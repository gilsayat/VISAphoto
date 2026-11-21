using DlibDotNet;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;

namespace USVisaPhotoAutomator.Services
{
    public enum EdgeFeatherMode
    {
        AutoSmart,
        Sharp,
        Balanced,
        Soft
    }

    public sealed class VisaPhotoProcessor : IDisposable
    {
        private readonly string _onnxModelPath;
        private readonly string _dlibModelPath;

        private readonly object _modelLock = new object();
        private InferenceSession? _session;
        private FrontalFaceDetector? _detector;
        private ShapePredictor? _predictor;
        private bool _isDisposed;

        public VisaPhotoProcessor(string modelsDirectory)
        {
            _onnxModelPath = Path.Combine(modelsDirectory, "u2net.onnx");
            _dlibModelPath = Path.Combine(modelsDirectory, "shape_predictor_68_face_landmarks.dat");
        }

        public void PreWarmModels()
        {
            EnsureModelsLoaded();
        }

        private void EnsureModelsLoaded()
        {
            if (_isDisposed) throw new ObjectDisposedException(nameof(VisaPhotoProcessor));

            if (_session != null && _detector != null && _predictor != null)
                return;

            lock (_modelLock)
            {
                if (_session != null && _detector != null && _predictor != null)
                    return;

                if (!File.Exists(_onnxModelPath))
                    throw new FileNotFoundException("Model Missing: 'u2net.onnx' not found in Models folder.");
                if (!File.Exists(_dlibModelPath))
                    throw new FileNotFoundException("Model Missing: 'shape_predictor_68_face_landmarks.dat' not found.");

                if (_session == null)
                {
                    var opts = new SessionOptions
                    {
                        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                        ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
                    };
                    _session = new InferenceSession(_onnxModelPath, opts);
                }

                _detector ??= Dlib.GetFrontalFaceDetector();
                _predictor ??= ShapePredictor.Deserialize(_dlibModelPath);
            }
        }

        public void ProcessVisaPhoto(
            string inputPath,
            string outputPath,
            string format,
            bool isPrintMode,
            EdgeFeatherMode edgeMode = EdgeFeatherMode.AutoSmart)
        {
            EnsureModelsLoaded();

            int targetSize = isPrintMode ? 1200 : 600;
            float dpi = isPrintMode ? 600f : 300f;

            using Bitmap originalImage = LoadOrientedBitmap(inputPath);
            if (originalImage.Width < 200 || originalImage.Height < 200)
                throw new Exception("Source image resolution is too low for a crisp visa photo.");

            using Bitmap workingImage = ResizeIfLarger(originalImage, 1600);

            // 1. Run U2-Net coarse portrait segmentation
            using Bitmap coarseMask = GetCoarsePortraitMask(workingImage);
            using Bitmap quickWhiteBg = ApplyPureWhiteBackground(workingImage, coarseMask);

            // 2. Facial landmark detection
            string tempFile = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".bmp");
            quickWhiteBg.Save(tempFile, ImageFormat.Bmp);

            int hairTopY, chinBottomY, browLineY;
            float eyeLineY, faceCenterX;

            try
            {
                using var dlibImg = Dlib.LoadImage<RgbPixel>(tempFile);

                var faces = _detector!.Operator(dlibImg, 0);
                if (faces.Length == 0)
                    faces = _detector.Operator(dlibImg, 1);

                if (faces.Length == 0)
                    throw new Exception("No face detected. Ensure lighting is even and the subject is looking forward.");
                if (faces.Length > 1)
                    throw new Exception($"Multiple faces detected ({faces.Length}). Visa photos must contain exactly 1 person.");

                var face = faces.OrderByDescending(r => r.Width * r.Height).First();
                using var shape = _predictor!.Detect(dlibImg, face);

                browLineY = (int)face.Top;
                chinBottomY = Math.Max((int)shape.GetPart(8).Y,
                              Math.Max((int)shape.GetPart(7).Y, (int)shape.GetPart(9).Y));

                float leftEyeX = 0, leftEyeY = 0, rightEyeX = 0, rightEyeY = 0;
                for (uint i = 36; i <= 41; i++)
                {
                    leftEyeX += shape.GetPart(i).X;
                    leftEyeY += shape.GetPart(i).Y;
                }
                for (uint i = 42; i <= 47; i++)
                {
                    rightEyeX += shape.GetPart(i).X;
                    rightEyeY += shape.GetPart(i).Y;
                }

                leftEyeX /= 6.0f;
                leftEyeY /= 6.0f;
                rightEyeX /= 6.0f;
                rightEyeY /= 6.0f;

                eyeLineY = (leftEyeY + rightEyeY) / 2.0f;
                float noseBridgeX = shape.GetPart(27).X;
                faceCenterX = (((leftEyeX + rightEyeX) / 2.0f) + noseBridgeX) / 2.0f;

                hairTopY = DetectHairTopY(coarseMask, (int)face.Left, (int)face.Right, browLineY, chinBottomY, faceCenterX);
            }
            finally
            {
                if (File.Exists(tempFile)) File.Delete(tempFile);
            }

            // 3. Landmark-aware Guided Filter refinement:
            //    Keeps ears/jaw/collar sharp while feathering hair according to EdgeFeatherMode
            using Bitmap refinedMask = RefineMaskWithGuidedFilter(workingImage, coarseMask, browLineY, edgeMode);
            using Bitmap whiteBgImage = ApplyPureWhiteBackground(workingImage, refinedMask);

            int rawHeadHeight = chinBottomY - hairTopY;
            if (rawHeadHeight <= 20)
                throw new Exception("Could not measure valid head height from facial landmarks.");

            // 4. U.S. Department of State Framing Solver (Head: 50%-69%, Eyes from bottom: 56%-69%)
            float targetHeadRatio = 0.56f;
            float scale = (targetSize * targetHeadRatio) / rawHeadHeight;

            if (scale > 4.5f)
                throw new Exception("Face is too small/far away to yield a sharp 2x2\" visa photo.");

            float scaledHairTopY = hairTopY * scale;
            float scaledEyeLineY = eyeLineY * scale;
            float eyeOffsetFromHairTop = scaledEyeLineY - scaledHairTopY;

            float desiredEyeFromTop = targetSize * (1.0f - 0.60f);
            float desiredHairTopFromTop = desiredEyeFromTop - eyeOffsetFromHairTop;

            float minHairTopFromTop = targetSize * 0.07f;
            if (desiredHairTopFromTop < minHairTopFromTop)
            {
                desiredHairTopFromTop = minHairTopFromTop;
                desiredEyeFromTop = desiredHairTopFromTop + eyeOffsetFromHairTop;

                float eyeFromBottomRatio = (targetSize - desiredEyeFromTop) / targetSize;
                if (eyeFromBottomRatio < 0.565f && eyeOffsetFromHairTop > 1f)
                {
                    float maxAllowedEyeFromTop = targetSize * (1.0f - 0.57f);
                    float maxAllowedCrownToEye = maxAllowedEyeFromTop - minHairTopFromTop;
                    float adjustFactor = maxAllowedCrownToEye / eyeOffsetFromHairTop;
                    scale *= adjustFactor;
                    scaledEyeLineY = eyeLineY * scale;
                    desiredEyeFromTop = maxAllowedEyeFromTop;
                }
            }

            int offsetX = (int)Math.Round((targetSize / 2.0f) - (faceCenterX * scale));
            int offsetY = (int)Math.Round(desiredEyeFromTop - scaledEyeLineY);

            int scaledWidth = Math.Max(1, (int)Math.Round(whiteBgImage.Width * scale));
            int scaledHeight = Math.Max(1, (int)Math.Round(whiteBgImage.Height * scale));

            using Bitmap finalImage = new Bitmap(targetSize, targetSize, PixelFormat.Format24bppRgb);
            finalImage.SetResolution(dpi, dpi);

            using (Graphics g = Graphics.FromImage(finalImage))
            {
                g.Clear(System.Drawing.Color.White);
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.CompositingQuality = CompositingQuality.HighQuality;
                g.DrawImage(whiteBgImage, offsetX, offsetY, scaledWidth, scaledHeight);
            }

            // 5. Export
            if (string.Equals(format, "PNG", StringComparison.OrdinalIgnoreCase))
                finalImage.Save(outputPath, ImageFormat.Png);
            else if (isPrintMode)
                SaveJpegWithQuality(finalImage, outputPath, 98);
            else
                SaveUnder240KbJpeg(finalImage, outputPath);
        }

        private static int DetectHairTopY(Bitmap mask, int faceLeft, int faceRight, int faceTop, int chinBottom, float faceCenterX)
        {
            int browToChin = Math.Max(20, chinBottom - faceTop);

            int defaultHairTop = Math.Max(0, faceTop - (int)Math.Round(browToChin * 0.45f));
            int lowestAllowedY = Math.Max(0, faceTop - (int)Math.Round(browToChin * 0.95f));
            int highestAllowedY = Math.Max(0, faceTop - (int)Math.Round(browToChin * 0.35f));

            int faceWidth = Math.Max(20, faceRight - faceLeft);
            int scanLeft = Math.Clamp((int)Math.Round(faceCenterX - faceWidth * 0.30f), 0, mask.Width - 1);
            int scanRight = Math.Clamp((int)Math.Round(faceCenterX + faceWidth * 0.30f), scanLeft, mask.Width - 1);
            int startY = Math.Clamp(faceTop, 0, mask.Height - 1);

            var rect = new System.Drawing.Rectangle(0, 0, mask.Width, mask.Height);
            var maskData = mask.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

            int detectedTop = defaultHairTop;
            try
            {
                unsafe
                {
                    byte* basePtr = (byte*)maskData.Scan0;
                    int stride = maskData.Stride;
                    int emptyRowStreak = 0;
                    int lastHairY = startY;

                    for (int y = startY; y >= lowestAllowedY; y--)
                    {
                        byte* row = basePtr + (y * stride);
                        bool hasHairPixel = false;

                        for (int x = scanLeft; x <= scanRight; x++)
                        {
                            if (row[x * 3] >= 115)
                            {
                                hasHairPixel = true;
                                break;
                            }
                        }

                        if (hasHairPixel)
                        {
                            lastHairY = y;
                            emptyRowStreak = 0;
                        }
                        else
                        {
                            emptyRowStreak++;
                            if (emptyRowStreak >= 3)
                                break;
                        }
                    }

                    detectedTop = Math.Clamp(lastHairY, lowestAllowedY, highestAllowedY);
                }
            }
            finally
            {
                mask.UnlockBits(maskData);
            }

            return detectedTop;
        }

        private Bitmap GetCoarsePortraitMask(Bitmap sourceImage)
        {
            const int dim = 320;
            using var resized = new Bitmap(dim, dim, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(resized))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.DrawImage(sourceImage, 0, 0, dim, dim);
            }

            var tensor = new DenseTensor<float>(new[] { 1, 3, dim, dim });
            var rect = new System.Drawing.Rectangle(0, 0, dim, dim);
            var bmpData = resized.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

            try
            {
                unsafe
                {
                    byte* basePtr = (byte*)bmpData.Scan0;
                    int stride = bmpData.Stride;
                    for (int y = 0; y < dim; y++)
                    {
                        byte* row = basePtr + (y * stride);
                        for (int x = 0; x < dim; x++)
                        {
                            byte b = row[x * 3];
                            byte gVal = row[x * 3 + 1];
                            byte r = row[x * 3 + 2];
                            tensor[0, 0, y, x] = (r / 255f - 0.485f) / 0.229f;
                            tensor[0, 1, y, x] = (gVal / 255f - 0.456f) / 0.224f;
                            tensor[0, 2, y, x] = (b / 255f - 0.406f) / 0.225f;
                        }
                    }
                }
            }
            finally { resized.UnlockBits(bmpData); }

            string inputName = _session!.InputMetadata.Keys.FirstOrDefault() ?? "input.1";
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensor) };

            using var results = _session.Run(inputs);
            var outTensor = results.First().AsTensor<float>();

            float min = float.MaxValue, max = float.MinValue;
            foreach (float val in outTensor)
            {
                if (val < min) min = val;
                if (val > max) max = val;
            }
            float range = max - min;

            using var smallMask = new Bitmap(dim, dim, PixelFormat.Format24bppRgb);
            var maskData = smallMask.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);

            try
            {
                unsafe
                {
                    byte* basePtr = (byte*)maskData.Scan0;
                    int stride = maskData.Stride;
                    for (int y = 0; y < dim; y++)
                    {
                        byte* row = basePtr + (y * stride);
                        for (int x = 0; x < dim; x++)
                        {
                            float val = outTensor[0, 0, y, x];
                            float norm = range > 1e-6f ? Math.Clamp((val - min) / range, 0f, 1f) : 0f;
                            byte m = (byte)Math.Round(norm * 255f);
                            row[x * 3] = m;
                            row[x * 3 + 1] = m;
                            row[x * 3 + 2] = m;
                        }
                    }
                }
            }
            finally { smallMask.UnlockBits(maskData); }

            var upscaledMask = new Bitmap(sourceImage.Width, sourceImage.Height, PixelFormat.Format24bppRgb);
            using (var gm = Graphics.FromImage(upscaledMask))
            {
                gm.InterpolationMode = InterpolationMode.HighQualityBicubic;
                gm.DrawImage(smallMask, 0, 0, sourceImage.Width, sourceImage.Height);
            }

            return upscaledMask;
        }

        private static Bitmap RefineMaskWithGuidedFilter(
            Bitmap guideImage,
            Bitmap coarseMask,
            int browLineY,
            EdgeFeatherMode mode)
        {
            int w = guideImage.Width;
            int h = guideImage.Height;
            int count = w * h;

            // Choose filter radius & regularization based on selected Edge mode
            int radius = mode switch
            {
                EdgeFeatherMode.Sharp => 3,
                EdgeFeatherMode.Soft => 7,
                _ => 5
            };

            float eps = mode switch
            {
                EdgeFeatherMode.Sharp => 0.0010f,
                EdgeFeatherMode.Soft => 0.0045f,
                _ => 0.0022f
            };

            float[] guide = new float[count];
            float[] p = new float[count];

            var rect = new System.Drawing.Rectangle(0, 0, w, h);
            var guideData = guideImage.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            var maskData = coarseMask.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

            double bgLumSum = 0;
            int bgPixelCount = 0;

            try
            {
                unsafe
                {
                    byte* gBase = (byte*)guideData.Scan0;
                    byte* mBase = (byte*)maskData.Scan0;
                    for (int y = 0; y < h; y++)
                    {
                        byte* gRow = gBase + (y * guideData.Stride);
                        byte* mRow = mBase + (y * maskData.Stride);
                        int rowOffset = y * w;
                        for (int x = 0; x < w; x++)
                        {
                            int idx = x * 3;
                            float lum = (0.0722f * gRow[idx] + 0.7152f * gRow[idx + 1] + 0.2126f * gRow[idx + 2]) / 255f;
                            float mVal = mRow[idx] / 255f;

                            guide[rowOffset + x] = lum;
                            p[rowOffset + x] = mVal;

                            if (mVal < 0.05f)
                            {
                                bgLumSum += lum;
                                bgPixelCount++;
                            }
                        }
                    }
                }
            }
            finally
            {
                guideImage.UnlockBits(guideData);
                coarseMask.UnlockBits(maskData);
            }

            float avgBgLum = bgPixelCount > 0 ? (float)(bgLumSum / bgPixelCount) : 0.5f;

            // Define thresholds for the hair zone (above browLineY) vs ear/jaw/collar zone (below browLineY)
            float hairLow, hairHigh, earLow, earHigh;

            switch (mode)
            {
                case EdgeFeatherMode.Sharp:
                    hairLow = 0.34f; hairHigh = 0.66f;
                    earLow = 0.38f;  earHigh = 0.62f;
                    break;
                case EdgeFeatherMode.Soft:
                    hairLow = 0.12f; hairHigh = 0.88f;
                    earLow = 0.26f;  earHigh = 0.74f;
                    break;
                case EdgeFeatherMode.Balanced:
                    hairLow = 0.22f; hairHigh = 0.78f;
                    earLow = 0.32f;  earHigh = 0.68f;
                    break;
                case EdgeFeatherMode.AutoSmart:
                default:
                    // Bright original background -> softer hair feathering without halo risk
                    // Dark original background -> tighter hair cutoff to prevent dark bleed
                    float brightnessFactor = Math.Clamp(avgBgLum, 0.2f, 0.9f);
                    float hairSpread = 0.26f + (brightnessFactor * 0.12f); // 0.28..0.37 half-width
                    hairLow = 0.50f - hairSpread;
                    hairHigh = 0.50f + hairSpread;

                    // Ears & collar always stay crisp
                    earLow = 0.33f;
                    earHigh = 0.67f;
                    break;
            }

            float[] ii = new float[count];
            float[] ip = new float[count];
            for (int i = 0; i < count; i++)
            {
                ii[i] = guide[i] * guide[i];
                ip[i] = guide[i] * p[i];
            }

            float[] meanI = BoxFilter(guide, w, h, radius);
            float[] meanP = BoxFilter(p, w, h, radius);
            float[] meanII = BoxFilter(ii, w, h, radius);
            float[] meanIP = BoxFilter(ip, w, h, radius);

            float[] a = new float[count];
            float[] b = new float[count];
            for (int i = 0; i < count; i++)
            {
                float varI = meanII[i] - meanI[i] * meanI[i];
                float covIP = meanIP[i] - meanI[i] * meanP[i];
                a[i] = covIP / (varI + eps);
                b[i] = meanP[i] - a[i] * meanI[i];
            }

            float[] meanA = BoxFilter(a, w, h, radius);
            float[] meanB = BoxFilter(b, w, h, radius);

            var refined = new Bitmap(w, h, PixelFormat.Format24bppRgb);
            var outData = refined.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);

            // Smooth vertical transition zone around browLineY so hair-to-ear sharpness blends seamlessly
            float transitionBand = Math.Max(20f, h * 0.04f);

            try
            {
                unsafe
                {
                    byte* oBase = (byte*)outData.Scan0;
                    for (int y = 0; y < h; y++)
                    {
                        byte* oRow = oBase + (y * outData.Stride);
                        int rowOffset = y * w;

                        // blend = 0 in hair region (above brows), 1 in ear/collar region (below brows)
                        float blend = Math.Clamp((y - (browLineY - transitionBand * 0.5f)) / transitionBand, 0f, 1f);
                        float low = hairLow + (earLow - hairLow) * blend;
                        float high = hairHigh + (earHigh - hairHigh) * blend;

                        for (int x = 0; x < w; x++)
                        {
                            int i = rowOffset + x;
                            float q = Math.Clamp(meanA[i] * guide[i] + meanB[i], 0f, 1f);

                            float alpha;
                            if (q <= low) alpha = 0f;
                            else if (q >= high) alpha = 1f;
                            else
                            {
                                float t = (q - low) / (high - low);
                                alpha = t * t * (3f - 2f * t);
                            }

                            byte m = (byte)Math.Round(alpha * 255f);
                            int idx = x * 3;
                            oRow[idx] = m;
                            oRow[idx + 1] = m;
                            oRow[idx + 2] = m;
                        }
                    }
                }
            }
            finally { refined.UnlockBits(outData); }

            return refined;
        }

        private static float[] BoxFilter(float[] src, int w, int h, int r)
        {
            int stride = w + 1;
            double[] integral = new double[stride * (h + 1)];

            for (int y = 0; y < h; y++)
            {
                double rowSum = 0.0;
                int srcRow = y * w;
                int prevIntRow = y * stride;
                int currIntRow = (y + 1) * stride;
                for (int x = 0; x < w; x++)
                {
                    rowSum += src[srcRow + x];
                    integral[currIntRow + x + 1] = integral[prevIntRow + x + 1] + rowSum;
                }
            }

            float[] dst = new float[w * h];
            for (int y = 0; y < h; y++)
            {
                int y0 = Math.Max(0, y - r);
                int y1 = Math.Min(h - 1, y + r);
                int r0 = y0 * stride;
                int r1 = (y1 + 1) * stride;
                int dstRow = y * w;

                for (int x = 0; x < w; x++)
                {
                    int x0 = Math.Max(0, x - r);
                    int x1 = Math.Min(w - 1, x + r);
                    double sum = integral[r1 + x1 + 1] - integral[r0 + x1 + 1] - integral[r1 + x0] + integral[r0 + x0];
                    int area = (x1 - x0 + 1) * (y1 - y0 + 1);
                    dst[dstRow + x] = (float)(sum / area);
                }
            }

            return dst;
        }

        private static Bitmap ApplyPureWhiteBackground(Bitmap image, Bitmap mask)
        {
            int width = image.Width;
            int height = image.Height;
            var result = new Bitmap(width, height, PixelFormat.Format24bppRgb);

            using var srcCopy = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            using var maskCopy = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            using (var g1 = Graphics.FromImage(srcCopy)) g1.DrawImage(image, 0, 0, width, height);
            using (var g2 = Graphics.FromImage(maskCopy)) g2.DrawImage(mask, 0, 0, width, height);

            var rect = new System.Drawing.Rectangle(0, 0, width, height);
            var srcData = srcCopy.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            var mskData = maskCopy.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            var dstData = result.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);

            try
            {
                unsafe
                {
                    byte* srcBase = (byte*)srcData.Scan0;
                    byte* mskBase = (byte*)mskData.Scan0;
                    byte* dstBase = (byte*)dstData.Scan0;

                    for (int y = 0; y < height; y++)
                    {
                        byte* sRow = srcBase + (y * srcData.Stride);
                        byte* mRow = mskBase + (y * mskData.Stride);
                        byte* dRow = dstBase + (y * dstData.Stride);

                        for (int x = 0; x < width; x++)
                        {
                            int idx = x * 3;
                            float alpha = mRow[idx] / 255f;

                            if (alpha <= 0.01f)
                            {
                                dRow[idx] = 255;
                                dRow[idx + 1] = 255;
                                dRow[idx + 2] = 255;
                            }
                            else if (alpha >= 0.99f)
                            {
                                dRow[idx] = sRow[idx];
                                dRow[idx + 1] = sRow[idx + 1];
                                dRow[idx + 2] = sRow[idx + 2];
                            }
                            else
                            {
                                float cleanAlpha = alpha * alpha;
                                dRow[idx] = (byte)Math.Clamp(Math.Round(sRow[idx] * cleanAlpha + 255f * (1f - cleanAlpha)), 0, 255);
                                dRow[idx + 1] = (byte)Math.Clamp(Math.Round(sRow[idx + 1] * cleanAlpha + 255f * (1f - cleanAlpha)), 0, 255);
                                dRow[idx + 2] = (byte)Math.Clamp(Math.Round(sRow[idx + 2] * cleanAlpha + 255f * (1f - cleanAlpha)), 0, 255);
                            }
                        }
                    }
                }
            }
            finally
            {
                srcCopy.UnlockBits(srcData);
                maskCopy.UnlockBits(mskData);
                result.UnlockBits(dstData);
            }
            return result;
        }

        private static Bitmap LoadOrientedBitmap(string inputPath)
        {
            using var fs = new FileStream(inputPath, FileMode.Open, FileAccess.Read);
            using var rawImg = System.Drawing.Image.FromStream(fs, useEmbeddedColorManagement: false, validateImageData: false);
            var bmp = new Bitmap(rawImg);

            const int orientationTagId = 0x0112;
            if (Array.IndexOf(rawImg.PropertyIdList, orientationTagId) >= 0)
            {
                var prop = rawImg.GetPropertyItem(orientationTagId);
                if (prop?.Value != null && prop.Value.Length >= 2)
                {
                    ushort orientation = BitConverter.ToUInt16(prop.Value, 0);
                    switch (orientation)
                    {
                        case 2: bmp.RotateFlip(RotateFlipType.RotateNoneFlipX); break;
                        case 3: bmp.RotateFlip(RotateFlipType.Rotate180FlipNone); break;
                        case 4: bmp.RotateFlip(RotateFlipType.Rotate180FlipX); break;
                        case 5: bmp.RotateFlip(RotateFlipType.Rotate90FlipX); break;
                        case 6: bmp.RotateFlip(RotateFlipType.Rotate90FlipNone); break;
                        case 7: bmp.RotateFlip(RotateFlipType.Rotate270FlipX); break;
                        case 8: bmp.RotateFlip(RotateFlipType.Rotate270FlipNone); break;
                    }
                }
            }

            return bmp;
        }

        private static Bitmap ResizeIfLarger(Bitmap source, int maxDim)
        {
            if (source.Width <= maxDim && source.Height <= maxDim)
                return new Bitmap(source);

            double ratio = Math.Min((double)maxDim / source.Width, (double)maxDim / source.Height);
            int w = Math.Max(1, (int)Math.Round(source.Width * ratio));
            int h = Math.Max(1, (int)Math.Round(source.Height * ratio));

            var resized = new Bitmap(w, h, PixelFormat.Format24bppRgb);
            using var g = Graphics.FromImage(resized);
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.DrawImage(source, 0, 0, w, h);
            return resized;
        }

        private static void SaveJpegWithQuality(Bitmap image, string outputPath, int quality)
        {
            var codec = ImageCodecInfo.GetImageEncoders().FirstOrDefault(c => c.FormatID == ImageFormat.Jpeg.Guid);
            if (codec == null) { image.Save(outputPath, ImageFormat.Jpeg); return; }

            using var encoderParams = new EncoderParameters(1);
            encoderParams.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, (long)quality);
            image.Save(outputPath, codec, encoderParams);
        }

        private static void SaveUnder240KbJpeg(Bitmap image, string outputPath)
        {
            var codec = ImageCodecInfo.GetImageEncoders().FirstOrDefault(c => c.FormatID == ImageFormat.Jpeg.Guid);
            if (codec == null) { image.Save(outputPath, ImageFormat.Jpeg); return; }

            int quality = 92;
            byte[] bestBytes = Array.Empty<byte>();

            while (quality >= 50)
            {
                using var ms = new MemoryStream();
                using var encoderParams = new EncoderParameters(1);
                encoderParams.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, (long)quality);
                image.Save(ms, codec, encoderParams);

                bestBytes = ms.ToArray();
                if (bestBytes.Length <= 240 * 1024) break;
                quality -= 4;
            }
            File.WriteAllBytes(outputPath, bestBytes);
        }

        public void Dispose()
        {
            if (_isDisposed) return;
            lock (_modelLock)
            {
                _session?.Dispose();
                _detector?.Dispose();
                _predictor?.Dispose();
                _session = null;
                _detector = null;
                _predictor = null;
                _isDisposed = true;
            }
        }
    }
}
