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
    public enum EdgeFeatherMode { AutoSmart, Sharp, Balanced, Soft }
    public enum PrintLayout { Digital2x2, Print4x6, Print4x8, Print5x10 }
    public enum BackgroundColorOption { White, OffWhite }

    public sealed class VisaPhotoProcessor : IDisposable
    {
        private readonly string _onnxModelPath;
        private readonly string _dlibModelPath;

        private readonly object _modelLock = new object();
        private InferenceSession? _session;
        private FrontalFaceDetector? _detector;
        private ShapePredictor? _predictor;
        private int _modelInputWidth = 320;
        private int _modelInputHeight = 320;
        private bool _isDisposed;

        public VisaPhotoProcessor(string modelsDirectory)
        {
            _onnxModelPath = Path.Combine(modelsDirectory, "u2net.onnx");
            _dlibModelPath = Path.Combine(modelsDirectory, "shape_predictor_68_face_landmarks.dat");
        }

        public void PreWarmModels() { EnsureModelsLoaded(); }

        private void EnsureModelsLoaded()
        {
            if (_isDisposed) throw new ObjectDisposedException(nameof(VisaPhotoProcessor));
            if (_session != null && _detector != null && _predictor != null) return;

            lock (_modelLock)
            {
                if (_session != null && _detector != null && _predictor != null) return;

                if (!File.Exists(_onnxModelPath)) throw new FileNotFoundException("Model Missing: 'u2net.onnx' not found in Models folder.");
                if (!File.Exists(_dlibModelPath)) throw new FileNotFoundException("Model Missing: 'shape_predictor_68_face_landmarks.dat' not found.");

                if (_session == null)
                {
                    var opts = new SessionOptions
                    {
                        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                        ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
                    };
                    _session = new InferenceSession(_onnxModelPath, opts);

                    var firstInput = _session.InputMetadata.Values.FirstOrDefault();
                    if (firstInput != null && firstInput.Dimensions.Length >= 4)
                    {
                        if (firstInput.Dimensions[2] > 0) _modelInputHeight = firstInput.Dimensions[2];
                        if (firstInput.Dimensions[3] > 0) _modelInputWidth = firstInput.Dimensions[3];
                    }
                }
                _detector ??= Dlib.GetFrontalFaceDetector();
                _predictor ??= ShapePredictor.Deserialize(_dlibModelPath);
            }
        }

        public void ProcessVisaPhoto(
            string inputPath, string outputPath, string format,
            PrintLayout layout, int dpi, BackgroundColorOption bgColor,
            EdgeFeatherMode edgeMode, bool fixLighting, double manualTiltAngle, double headZoomRatio)
        {
            EnsureModelsLoaded();
            int targetSize = 2 * dpi; 
            byte bgTarget = bgColor == BackgroundColorOption.OffWhite ? (byte)240 : (byte)255;

            using Bitmap originalImage = LoadOrientedBitmap(inputPath);
            if (originalImage.Width < 200 || originalImage.Height < 200)
                throw new Exception("Source image resolution is too low for a crisp visa photo.");

            Bitmap workingImage = ResizeIfLarger(originalImage, 2200);

            // 1. Manual Tilt Rotation
            Bitmap leveledImage = workingImage;
            if (Math.Abs(manualTiltAngle) > 0.1)
            {
                leveledImage = new Bitmap(workingImage.Width, workingImage.Height, PixelFormat.Format24bppRgb);
                leveledImage.SetResolution(workingImage.HorizontalResolution, workingImage.VerticalResolution);
                using (Graphics g = Graphics.FromImage(leveledImage))
                {
                    g.Clear(System.Drawing.Color.White);
                    g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    g.TranslateTransform(workingImage.Width / 2f, workingImage.Height / 2f);
                    g.RotateTransform((float)manualTiltAngle);
                    g.TranslateTransform(-workingImage.Width / 2f, -workingImage.Height / 2f);
                    g.DrawImage(workingImage, new System.Drawing.Point(0, 0));
                }
                workingImage.Dispose();
            }

            try
            {
                // 2. Landmark Detection & Safety
                string detectTempFile = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".bmp");
                leveledImage.Save(detectTempFile, ImageFormat.Bmp);

                int faceLeft, faceRight, browLineY, chinBottomY;
                float eyeLineY, faceCenterX;

                try
                {
                    using var dlibImg = Dlib.LoadImage<RgbPixel>(detectTempFile);
                    var faces = _detector!.Operator(dlibImg, 0);
                    if (faces.Length == 0) faces = _detector.Operator(dlibImg, 1);
                    if (faces.Length == 0) throw new Exception("No face detected. Ensure lighting is even and subject looks forward.");
                    if (faces.Length > 1) throw new Exception($"Multiple faces detected ({faces.Length}). Visa photos must contain exactly 1 person.");

                    var face = faces.OrderByDescending(r => r.Width * r.Height).First();
                    using var shape = _predictor!.Detect(dlibImg, face);

                    float mouthHeight = shape.GetPart(66).Y - shape.GetPart(62).Y;
                    float mouthWidth = shape.GetPart(64).X - shape.GetPart(60).X;
                    if (mouthWidth > 0 && (mouthHeight / mouthWidth) > 0.18f)
                        throw new Exception("Subject's mouth appears open or smiling. US Visa rules require a neutral expression.");

                    faceLeft = (int)face.Left; faceRight = (int)face.Right; browLineY = (int)face.Top;
                    chinBottomY = Math.Max((int)shape.GetPart(8).Y, Math.Max((int)shape.GetPart(7).Y, (int)shape.GetPart(9).Y));

                    float leftEyeX = 0, leftEyeY = 0, rightEyeX = 0, rightEyeY = 0;
                    for (uint i = 36; i <= 41; i++) { leftEyeX += shape.GetPart(i).X; leftEyeY += shape.GetPart(i).Y; }
                    for (uint i = 42; i <= 47; i++) { rightEyeX += shape.GetPart(i).X; rightEyeY += shape.GetPart(i).Y; }
                    leftEyeX /= 6.0f; leftEyeY /= 6.0f; rightEyeX /= 6.0f; rightEyeY /= 6.0f;

                    eyeLineY = (leftEyeY + rightEyeY) / 2.0f;
                    float noseBridgeX = shape.GetPart(27).X;
                    faceCenterX = (((leftEyeX + rightEyeX) / 2.0f) + noseBridgeX) / 2.0f;
                }
                finally { if (File.Exists(detectTempFile)) File.Delete(detectTempFile); }

                // 3. Face-Aware Lighting
                using Bitmap litImage = fixLighting
                    ? ApplyFaceAwareLightingBalance(leveledImage, faceLeft, faceRight, browLineY, chinBottomY)
                    : new Bitmap(leveledImage);

                // 4. Crop ROI
                int browToChin = Math.Max(30, chinBottomY - browLineY);
                int halfRoiWidth = (int)Math.Round(browToChin * 1.65f);
                int roiLeft = Math.Clamp((int)Math.Round(faceCenterX) - halfRoiWidth, 0, litImage.Width - 1);
                int roiRight = Math.Clamp((int)Math.Round(faceCenterX) + halfRoiWidth, roiLeft + 1, litImage.Width);
                int roiTop = Math.Clamp(browLineY - (int)Math.Round(browToChin * 1.35f), 0, litImage.Height - 1);
                int roiBottom = Math.Clamp(chinBottomY + (int)Math.Round(browToChin * 1.45f), roiTop + 1, litImage.Height);

                var roiRect = new System.Drawing.Rectangle(roiLeft, roiTop, roiRight - roiLeft, roiBottom - roiTop);
                using Bitmap roiImage = CropBitmap(litImage, roiRect);

                int roiFaceLeft = faceLeft - roiLeft, roiFaceRight = faceRight - roiLeft, roiBrowLineY = browLineY - roiTop, roiChinBottomY = chinBottomY - roiTop;
                float roiEyeLineY = eyeLineY - roiTop, roiFaceCenterX = faceCenterX - roiLeft;

                // 5. Matting & Color Decontamination
                using Bitmap coarseMask = GetCoarsePortraitMaskWithTTA(roiImage);
                int roiHairTopY = DetectHairTopY(coarseMask, roiFaceLeft, roiFaceRight, roiBrowLineY, roiChinBottomY, roiFaceCenterX);

                using Bitmap refinedAlpha = RefineAlphaWithLumaChromaGuidedFilter(roiImage, coarseMask, roiBrowLineY, edgeMode);
                using Bitmap decontaminatedBg = CompositeWithColorDecontamination(roiImage, refinedAlpha, bgTarget);

                int rawHeadHeight = roiChinBottomY - roiHairTopY;
                if (rawHeadHeight <= 20) throw new Exception("Could not measure valid head height.");

                // 6. Strict Zoom Framing (Head: 50%-69%, Eyes: 56%-69% from bottom)
                float scale = (targetSize * (float)headZoomRatio) / rawHeadHeight;
                if (scale > 4.5f) throw new Exception("Face is too small/far away to yield a sharp visa photo.");

                float scaledHairTopY = roiHairTopY * scale;
                float scaledEyeLineY = roiEyeLineY * scale;
                float eyeOffsetFromHairTop = scaledEyeLineY - scaledHairTopY;

                float desiredEyeFromTop = targetSize * (1.0f - 0.60f); // Default 60% from bottom
                float desiredHairTopFromTop = desiredEyeFromTop - eyeOffsetFromHairTop;
                float minHairTopFromTop = targetSize * 0.05f; // Hard minimum headroom

                if (desiredHairTopFromTop < minHairTopFromTop)
                {
                    desiredHairTopFromTop = minHairTopFromTop;
                    desiredEyeFromTop = desiredHairTopFromTop + eyeOffsetFromHairTop;
                    float maxEyeFromTop = targetSize * (1.0f - 0.56f); // 56% from bottom bounds
                    if (desiredEyeFromTop > maxEyeFromTop) desiredEyeFromTop = maxEyeFromTop;
                }

                int offsetX = (int)Math.Round((targetSize / 2.0f) - (roiFaceCenterX * scale));
                int offsetY = (int)Math.Round(desiredEyeFromTop - scaledEyeLineY);

                int scaledWidth = Math.Max(1, (int)Math.Round(decontaminatedBg.Width * scale));
                int scaledHeight = Math.Max(1, (int)Math.Round(decontaminatedBg.Height * scale));

                using Bitmap base2x2 = new Bitmap(targetSize, targetSize, PixelFormat.Format24bppRgb);
                base2x2.SetResolution(dpi, dpi);
                using (Graphics g = Graphics.FromImage(base2x2))
                {
                    g.Clear(System.Drawing.Color.FromArgb(bgTarget, bgTarget, bgTarget));
                    g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                    g.SmoothingMode = SmoothingMode.HighQuality;
                    g.DrawImage(decontaminatedBg, offsetX, offsetY, scaledWidth, scaledHeight);
                    
                    // Bleed-safe cut lines for print layouts
                    if (layout != PrintLayout.Digital2x2)
                        g.DrawRectangle(new Pen(System.Drawing.Color.Black, 2f), 4, 4, targetSize - 8, targetSize - 8);
                }

                // 7. Render Layout Sheets
                using Bitmap finalSheet = (layout == PrintLayout.Digital2x2) ? new Bitmap(base2x2) : GeneratePrintSheet(base2x2, layout, dpi, bgTarget);

                // 8. Export
                if (string.Equals(format, "PNG", StringComparison.OrdinalIgnoreCase))
                    finalSheet.Save(outputPath, ImageFormat.Png);
                else if (layout != PrintLayout.Digital2x2)
                    SaveJpegWithQuality(finalSheet, outputPath, 98); // High quality for prints
                else
                    SaveUnder240KbJpeg(finalSheet, outputPath); // Under 240KB for digital web upload
            }
            finally { leveledImage.Dispose(); }
        }

        private static Bitmap GeneratePrintSheet(Bitmap base2x2, PrintLayout layout, int dpi, byte bgTarget)
        {
            int targetSize = base2x2.Width;
            int w = 4 * dpi, h = 6 * dpi;
            if (layout == PrintLayout.Print4x8) h = 8 * dpi;
            if (layout == PrintLayout.Print5x10) { w = 5 * dpi; h = 10 * dpi; }
            
            var sheet = new Bitmap(w, h, PixelFormat.Format24bppRgb);
            sheet.SetResolution(dpi, dpi);
            using Graphics gs = Graphics.FromImage(sheet);
            gs.Clear(System.Drawing.Color.White);
            
            int rows = (layout == PrintLayout.Print4x6) ? 3 : 4;
            int startX = (layout == PrintLayout.Print5x10) ? (int)(0.5f * dpi) : 0;
            int startY = (layout == PrintLayout.Print5x10) ? (int)(1.0f * dpi) : 0;

            for (int y = 0; y < rows; y++)
                for (int x = 0; x < 2; x++)
                    gs.DrawImageUnscaled(base2x2, startX + x * targetSize, startY + y * targetSize);
            return sheet;
        }

        private static Bitmap ApplyFaceAwareLightingBalance(Bitmap source, int faceLeft, int faceRight, int browLineY, int chinBottomY)
        {
            int w = source.Width, h = source.Height;
            int sampleLeft = Math.Clamp(faceLeft + (faceRight - faceLeft) / 6, 0, w - 1);
            int sampleRight = Math.Clamp(faceRight - (faceRight - faceLeft) / 6, sampleLeft, w - 1);
            int sampleTop = Math.Clamp(browLineY, 0, h - 1);
            int sampleBottom = Math.Clamp(chinBottomY, sampleTop, h - 1);

            var rect = new System.Drawing.Rectangle(0, 0, w, h);
            using var srcCopy = new Bitmap(w, h, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(srcCopy)) g.DrawImage(source, 0, 0, w, h);
            var srcData = srcCopy.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            double faceLumSum = 0; int faceCount = 0;

            try
            {
                unsafe
                {
                    byte* basePtr = (byte*)srcData.Scan0; int stride = srcData.Stride;
                    for (int y = sampleTop; y <= sampleBottom; y++)
                    {
                        byte* row = basePtr + (y * stride);
                        for (int x = sampleLeft; x <= sampleRight; x++)
                        {
                            int idx = x * 3;
                            faceLumSum += 0.299f * (row[idx + 2] / 255f) + 0.587f * (row[idx + 1] / 255f) + 0.114f * (row[idx] / 255f);
                            faceCount++;
                        }
                    }
                }
            }
            finally { srcCopy.UnlockBits(srcData); }

            float avgFaceLum = faceCount > 0 ? (float)(faceLumSum / faceCount) : 0.56f;
            float gamma = Math.Clamp((float)(Math.Log(0.56) / Math.Log(Math.Max(0.08, Math.Clamp(avgFaceLum, 0.25f, 0.72f)))), 0.75f, 1.35f);
            if (Math.Abs(gamma - 1.0f) < 0.04f) return new Bitmap(source);

            byte[] lut = new byte[256];
            for (int i = 0; i < 256; i++)
            {
                double norm = i / 255.0;
                lut[i] = (byte)Math.Clamp(Math.Round((Math.Pow(norm, gamma) + 0.04 * (1.0 - norm) * (1.0 - norm)) * 255.0), 0, 255);
            }

            var output = new Bitmap(w, h, PixelFormat.Format24bppRgb);
            var outData = output.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
            var readData = srcCopy.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

            try
            {
                unsafe
                {
                    byte* sBase = (byte*)readData.Scan0; byte* dBase = (byte*)outData.Scan0;
                    for (int y = 0; y < h; y++)
                    {
                        byte* sRow = sBase + (y * readData.Stride); byte* dRow = dBase + (y * outData.Stride);
                        for (int x = 0; x < w; x++)
                        {
                            int idx = x * 3;
                            dRow[idx] = lut[sRow[idx]]; dRow[idx + 1] = lut[sRow[idx + 1]]; dRow[idx + 2] = lut[sRow[idx + 2]];
                        }
                    }
                }
            }
            finally { srcCopy.UnlockBits(readData); output.UnlockBits(outData); }
            return output;
        }

        private static Bitmap CropBitmap(Bitmap source, System.Drawing.Rectangle cropRect)
        {
            var dst = new Bitmap(cropRect.Width, cropRect.Height, PixelFormat.Format24bppRgb);
            using var g = Graphics.FromImage(dst);
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.DrawImage(source, new System.Drawing.Rectangle(0, 0, cropRect.Width, cropRect.Height), cropRect, GraphicsUnit.Pixel);
            return dst;
        }

        private static int DetectHairTopY(Bitmap mask, int faceLeft, int faceRight, int faceTop, int chinBottom, float faceCenterX)
        {
            int browToChin = Math.Max(20, chinBottom - faceTop);
            int lowestAllowedY = Math.Max(0, faceTop - (int)Math.Round(browToChin * 0.95f));
            int highestAllowedY = Math.Max(0, faceTop - (int)Math.Round(browToChin * 0.35f));
            int faceWidth = Math.Max(20, faceRight - faceLeft);
            int scanLeft = Math.Clamp((int)Math.Round(faceCenterX - faceWidth * 0.30f), 0, mask.Width - 1);
            int scanRight = Math.Clamp((int)Math.Round(faceCenterX + faceWidth * 0.30f), scanLeft, mask.Width - 1);

            var maskData = mask.LockBits(new System.Drawing.Rectangle(0, 0, mask.Width, mask.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            int lastHairY = Math.Clamp(faceTop, 0, mask.Height - 1);
            try
            {
                unsafe
                {
                    byte* basePtr = (byte*)maskData.Scan0; int emptyRowStreak = 0;
                    for (int y = lastHairY; y >= lowestAllowedY; y--)
                    {
                        byte* row = basePtr + (y * maskData.Stride);
                        bool hasHairPixel = false;
                        for (int x = scanLeft; x <= scanRight; x++) { if (row[x * 3] >= 115) { hasHairPixel = true; break; } }
                        if (hasHairPixel) { lastHairY = y; emptyRowStreak = 0; }
                        else { emptyRowStreak++; if (emptyRowStreak >= 3) break; }
                    }
                }
            }
            finally { mask.UnlockBits(maskData); }
            return Math.Clamp(lastHairY, lowestAllowedY, highestAllowedY);
        }

        private Bitmap GetCoarsePortraitMaskWithTTA(Bitmap sourceImage)
        {
            float[] maskNormal = RunSingleOnnxMask(sourceImage, _modelInputWidth, _modelInputHeight);
            using var flippedSource = new Bitmap(sourceImage);
            flippedSource.RotateFlip(RotateFlipType.RotateNoneFlipX);
            float[] maskFlipped = RunSingleOnnxMask(flippedSource, _modelInputWidth, _modelInputHeight);

            using var smallMask = new Bitmap(_modelInputWidth, _modelInputHeight, PixelFormat.Format24bppRgb);
            var maskData = smallMask.LockBits(new System.Drawing.Rectangle(0, 0, _modelInputWidth, _modelInputHeight), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
            try
            {
                unsafe
                {
                    byte* basePtr = (byte*)maskData.Scan0;
                    for (int y = 0; y < _modelInputHeight; y++)
                    {
                        byte* row = basePtr + (y * maskData.Stride); int rowIdx = y * _modelInputWidth;
                        for (int x = 0; x < _modelInputWidth; x++)
                        {
                            byte m = (byte)Math.Round(Math.Clamp(0.5f * (maskNormal[rowIdx + x] + maskFlipped[rowIdx + (_modelInputWidth - 1 - x)]), 0f, 1f) * 255f);
                            row[x * 3] = m; row[x * 3 + 1] = m; row[x * 3 + 2] = m;
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

        private float[] RunSingleOnnxMask(Bitmap sourceImage, int dimW, int dimH)
        {
            using var resized = new Bitmap(dimW, dimH, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(resized)) { g.InterpolationMode = InterpolationMode.HighQualityBicubic; g.DrawImage(sourceImage, 0, 0, dimW, dimH); }
            var tensor = new DenseTensor<float>(new[] { 1, 3, dimH, dimW });
            var bmpData = resized.LockBits(new System.Drawing.Rectangle(0, 0, dimW, dimH), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            try
            {
                unsafe
                {
                    byte* basePtr = (byte*)bmpData.Scan0;
                    for (int y = 0; y < dimH; y++)
                    {
                        byte* row = basePtr + (y * bmpData.Stride);
                        for (int x = 0; x < dimW; x++)
                        {
                            tensor[0, 0, y, x] = (row[x * 3 + 2] / 255f - 0.485f) / 0.229f;
                            tensor[0, 1, y, x] = (row[x * 3 + 1] / 255f - 0.456f) / 0.224f;
                            tensor[0, 2, y, x] = (row[x * 3] / 255f - 0.406f) / 0.225f;
                        }
                    }
                }
            }
            finally { resized.UnlockBits(bmpData); }

            using var results = _session.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_session!.InputMetadata.Keys.FirstOrDefault() ?? "input.1", tensor) });
            var outTensor = results.First().AsTensor<float>();

            float min = float.MaxValue, max = float.MinValue;
            foreach (float val in outTensor) { if (val < min) min = val; if (val > max) max = val; }
            float range = max - min;
            float[] normOut = new float[dimW * dimH];
            for (int y = 0; y < dimH; y++)
            {
                int rowIdx = y * dimW;
                for (int x = 0; x < dimW; x++)
                {
                    normOut[rowIdx + x] = range > 1e-6f ? Math.Clamp((outTensor[0, 0, y, x] - min) / range, 0f, 1f) : 0f;
                }
            }
            return normOut;
        }

        private static Bitmap RefineAlphaWithLumaChromaGuidedFilter(Bitmap guideImage, Bitmap coarseMask, int browLineY, EdgeFeatherMode mode)
        {
            int w = guideImage.Width, h = guideImage.Height, count = w * h;
            int radius = mode switch { EdgeFeatherMode.Sharp => 3, EdgeFeatherMode.Soft => 7, _ => 5 };
            float eps = mode switch { EdgeFeatherMode.Sharp => 0.0008f, EdgeFeatherMode.Soft => 0.0035f, _ => 0.0016f };

            float[] g1 = new float[count], g2 = new float[count], p = new float[count];
            var guideData = guideImage.LockBits(new System.Drawing.Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            var maskData = coarseMask.LockBits(new System.Drawing.Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            try
            {
                unsafe
                {
                    byte* gBase = (byte*)guideData.Scan0, mBase = (byte*)maskData.Scan0;
                    for (int y = 0; y < h; y++)
                    {
                        byte* gRow = gBase + (y * guideData.Stride), mRow = mBase + (y * maskData.Stride); int rowOffset = y * w;
                        for (int x = 0; x < w; x++)
                        {
                            int idx = x * 3;
                            float b = gRow[idx] / 255f, g = gRow[idx + 1] / 255f, r = gRow[idx + 2] / 255f;
                            g1[rowOffset + x] = 0.299f * r + 0.587f * g + 0.114f * b;
                            g2[rowOffset + x] = (r - b) * 0.5f + 0.5f;
                            p[rowOffset + x] = mRow[idx] / 255f;
                        }
                    }
                }
            }
            finally { guideImage.UnlockBits(guideData); coarseMask.UnlockBits(maskData); }

            float[] g1g1 = new float[count], g2g2 = new float[count], g1g2 = new float[count], g1p = new float[count], g2p = new float[count];
            for (int i = 0; i < count; i++)
            {
                g1g1[i] = g1[i] * g1[i]; g2g2[i] = g2[i] * g2[i]; g1g2[i] = g1[i] * g2[i];
                g1p[i] = g1[i] * p[i]; g2p[i] = g2[i] * p[i];
            }

            float[] meanG1 = BoxFilter(g1, w, h, radius), meanG2 = BoxFilter(g2, w, h, radius), meanP = BoxFilter(p, w, h, radius);
            float[] meanG1G1 = BoxFilter(g1g1, w, h, radius), meanG2G2 = BoxFilter(g2g2, w, h, radius), meanG1G2 = BoxFilter(g1g2, w, h, radius);
            float[] meanG1P = BoxFilter(g1p, w, h, radius), meanG2P = BoxFilter(g2p, w, h, radius);
            float[] a1 = new float[count], a2 = new float[count], bCoef = new float[count];

            for (int i = 0; i < count; i++)
            {
                float var1 = meanG1G1[i] - meanG1[i] * meanG1[i] + eps, var2 = meanG2G2[i] - meanG2[i] * meanG2[i] + eps;
                float cov12 = meanG1G2[i] - meanG1[i] * meanG2[i];
                float det = var1 * var2 - cov12 * cov12; if (Math.Abs(det) < 1e-10f) det = 1e-10f;
                a1[i] = (var2 * (meanG1P[i] - meanG1[i] * meanP[i]) - cov12 * (meanG2P[i] - meanG2[i] * meanP[i])) / det;
                a2[i] = (var1 * (meanG2P[i] - meanG2[i] * meanP[i]) - cov12 * (meanG1P[i] - meanG1[i] * meanP[i])) / det;
                bCoef[i] = meanP[i] - a1[i] * meanG1[i] - a2[i] * meanG2[i];
            }

            float[] meanA1 = BoxFilter(a1, w, h, radius), meanA2 = BoxFilter(a2, w, h, radius), meanB = BoxFilter(bCoef, w, h, radius);
            float hairLow, hairHigh, earLow, earHigh;
            switch (mode)
            {
                case EdgeFeatherMode.Sharp: hairLow = 0.28f; hairHigh = 0.72f; earLow = 0.32f; earHigh = 0.68f; break;
                case EdgeFeatherMode.Soft: hairLow = 0.10f; hairHigh = 0.90f; earLow = 0.18f; earHigh = 0.82f; break;
                case EdgeFeatherMode.Balanced: hairLow = 0.18f; hairHigh = 0.82f; earLow = 0.24f; earHigh = 0.76f; break;
                default: hairLow = 0.15f; hairHigh = 0.85f; earLow = 0.25f; earHigh = 0.75f; break;
            }

            float[] rawAlpha = new float[count]; float transitionBand = Math.Max(24f, h * 0.05f);
            for (int y = 0; y < h; y++)
            {
                int rowOffset = y * w; float blend = Math.Clamp((y - (browLineY - transitionBand * 0.5f)) / transitionBand, 0f, 1f);
                float low = hairLow + (earLow - hairLow) * blend, high = hairHigh + (earHigh - hairHigh) * blend;
                for (int x = 0; x < w; x++)
                {
                    int i = rowOffset + x; float q = Math.Clamp(meanA1[i] * g1[i] + meanA2[i] * g2[i] + meanB[i], 0f, 1f);
                    if (q <= low) rawAlpha[i] = 0f;
                    else if (q >= high) rawAlpha[i] = 1f;
                    else { float t = (q - low) / (high - low); rawAlpha[i] = t * t * (3f - 2f * t); }
                }
            }

            float[] smoothAlpha = BoxFilter(rawAlpha, w, h, 1);
            var refined = new Bitmap(w, h, PixelFormat.Format24bppRgb);
            var outData = refined.LockBits(new System.Drawing.Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
            try
            {
                unsafe
                {
                    byte* oBase = (byte*)outData.Scan0;
                    for (int y = 0; y < h; y++)
                    {
                        byte* oRow = oBase + (y * outData.Stride); int rowOffset = y * w;
                        for (int x = 0; x < w; x++)
                        {
                            byte m = (byte)Math.Round(Math.Clamp(smoothAlpha[rowOffset + x], 0f, 1f) * 255f);
                            int idx = x * 3; oRow[idx] = m; oRow[idx + 1] = m; oRow[idx + 2] = m;
                        }
                    }
                }
            }
            finally { refined.UnlockBits(outData); }
            return refined;
        }

        private static Bitmap CompositeWithColorDecontamination(Bitmap image, Bitmap alphaMask, byte bgTarget)
        {
            int w = image.Width, h = image.Height, count = w * h;
            float[] srcB = new float[count], srcG = new float[count], srcR = new float[count], alpha = new float[count];
            float[] bgB = new float[count], bgG = new float[count], bgR = new float[count], bgW = new float[count];
            double globalBgB = 0, globalBgG = 0, globalBgR = 0; int globalBgCount = 0;

            var rect = new System.Drawing.Rectangle(0, 0, w, h);
            var srcData = image.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            var mskData = alphaMask.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

            try
            {
                unsafe
                {
                    byte* sBase = (byte*)srcData.Scan0, mBase = (byte*)mskData.Scan0;
                    for (int y = 0; y < h; y++)
                    {
                        byte* sRow = sBase + (y * srcData.Stride), mRow = mBase + (y * mskData.Stride); int rowOffset = y * w;
                        for (int x = 0; x < w; x++)
                        {
                            int i = rowOffset + x; int idx = x * 3;
                            float b = sRow[idx], g = sRow[idx + 1], r = sRow[idx + 2], a = mRow[idx] / 255f;
                            srcB[i] = b; srcG[i] = g; srcR[i] = r; alpha[i] = a;

                            if (a <= 0.04f)
                            {
                                bgB[i] = b; bgG[i] = g; bgR[i] = r; bgW[i] = 1f;
                                globalBgB += b; globalBgG += g; globalBgR += r; globalBgCount++;
                            }
                        }
                    }
                }
            }
            finally { image.UnlockBits(srcData); alphaMask.UnlockBits(mskData); }

            float fallbackB = globalBgCount > 0 ? (float)(globalBgB / globalBgCount) : 200f;
            float fallbackG = globalBgCount > 0 ? (float)(globalBgG / globalBgCount) : 200f;
            float fallbackR = globalBgCount > 0 ? (float)(globalBgR / globalBgCount) : 200f;

            float[] sumBgB = BoxFilter(bgB, w, h, 28), sumBgG = BoxFilter(bgG, w, h, 28);
            float[] sumBgR = BoxFilter(bgR, w, h, 28), sumBgW = BoxFilter(bgW, w, h, 28);

            var result = new Bitmap(w, h, PixelFormat.Format24bppRgb);
            var dstData = result.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
            try
            {
                unsafe
                {
                    byte* dBase = (byte*)dstData.Scan0;
                    for (int y = 0; y < h; y++)
                    {
                        byte* dRow = dBase + (y * dstData.Stride); int rowOffset = y * w;
                        for (int x = 0; x < w; x++)
                        {
                            int i = rowOffset + x; int idx = x * 3; float a = alpha[i];

                            if (a <= 0.005f) { dRow[idx] = bgTarget; dRow[idx + 1] = bgTarget; dRow[idx + 2] = bgTarget; }
                            else if (a >= 0.995f) { dRow[idx] = (byte)Math.Round(srcB[i]); dRow[idx + 1] = (byte)Math.Round(srcG[i]); dRow[idx + 2] = (byte)Math.Round(srcR[i]); }
                            else
                            {
                                float weight = sumBgW[i];
                                float localB = weight > 1e-4f ? (sumBgB[i] / weight) : fallbackB;
                                float localG = weight > 1e-4f ? (sumBgG[i] / weight) : fallbackG;
                                float localR = weight > 1e-4f ? (sumBgR[i] / weight) : fallbackR;
                                float oneMinusA = 1f - a;
                                dRow[idx] = (byte)Math.Clamp(Math.Round(srcB[i] + oneMinusA * (bgTarget - localB)), 0, 255);
                                dRow[idx + 1] = (byte)Math.Clamp(Math.Round(srcG[i] + oneMinusA * (bgTarget - localG)), 0, 255);
                                dRow[idx + 2] = (byte)Math.Clamp(Math.Round(srcR[i] + oneMinusA * (bgTarget - localR)), 0, 255);
                            }
                        }
                    }
                }
            }
            finally { result.UnlockBits(dstData); }
            return result;
        }

        private static float[] BoxFilter(float[] src, int w, int h, int r)
        {
            int stride = w + 1; double[] integral = new double[stride * (h + 1)];
            for (int y = 0; y < h; y++)
            {
                double rowSum = 0.0; int srcRow = y * w, prevIntRow = y * stride, currIntRow = (y + 1) * stride;
                for (int x = 0; x < w; x++)
                {
                    rowSum += src[srcRow + x];
                    integral[currIntRow + x + 1] = integral[prevIntRow + x + 1] + rowSum;
                }
            }
            float[] dst = new float[w * h];
            for (int y = 0; y < h; y++)
            {
                int y0 = Math.Max(0, y - r), y1 = Math.Min(h - 1, y + r);
                int r0 = y0 * stride, r1 = (y1 + 1) * stride, dstRow = y * w;
                for (int x = 0; x < w; x++)
                {
                    int x0 = Math.Max(0, x - r), x1 = Math.Min(w - 1, x + r);
                    double sum = integral[r1 + x1 + 1] - integral[r0 + x1 + 1] - integral[r1 + x0] + integral[r0 + x0];
                    dst[dstRow + x] = (float)(sum / ((x1 - x0 + 1) * (y1 - y0 + 1)));
                }
            }
            return dst;
        }

        private static Bitmap LoadOrientedBitmap(string inputPath)
        {
            using var fs = new FileStream(inputPath, FileMode.Open, FileAccess.Read);
            using var rawImg = System.Drawing.Image.FromStream(fs, useEmbeddedColorManagement: false, validateImageData: false);
            var bmp = new Bitmap(rawImg);
            if (Array.IndexOf(rawImg.PropertyIdList, 0x0112) >= 0)
            {
                var prop = rawImg.GetPropertyItem(0x0112);
                if (prop?.Value != null && prop.Value.Length >= 2)
                {
                    switch (BitConverter.ToUInt16(prop.Value, 0))
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
            if (source.Width <= maxDim && source.Height <= maxDim) return new Bitmap(source);
            double ratio = Math.Min((double)maxDim / source.Width, (double)maxDim / source.Height);
            int w = Math.Max(1, (int)Math.Round(source.Width * ratio)), h = Math.Max(1, (int)Math.Round(source.Height * ratio));
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
            int quality = 92; byte[] bestBytes = Array.Empty<byte>();
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
            lock (_modelLock) { _session?.Dispose(); _detector?.Dispose(); _predictor?.Dispose(); _session = null; _detector = null; _predictor = null; _isDisposed = true; }
        }
    }
}
