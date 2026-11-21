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

        public void ProcessVisaPhoto(string inputPath, string outputPath, string format, bool isPrintMode)
        {
            EnsureModelsLoaded();

            // 1200x1200 @ 600 DPI = 2.0x2.0 in | 600x600 @ 300 DPI = 2.0x2.0 in
            int targetSize = isPrintMode ? 1200 : 600;
            float dpi = isPrintMode ? 600f : 300f;

            using Bitmap originalImage = LoadOrientedBitmap(inputPath);
            if (originalImage.Width < 200 || originalImage.Height < 200)
                throw new Exception("Source image resolution is too low for a crisp visa photo.");

            using Bitmap workingImage = ResizeIfLarger(originalImage, 1600);

            // 1. Background segmentation & pure white (#FFFFFF) replacement
            using Bitmap alphaMask = GetPortraitAlphaMask(workingImage);
            using Bitmap whiteBgImage = ApplyPureWhiteBackground(workingImage, alphaMask);

            // 2. Facial landmark detection with multi-scale fallback
            string tempFile = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".bmp");
            whiteBgImage.Save(tempFile, ImageFormat.Bmp);

            int hairTopY, chinBottomY;
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

                // Chin bottom (points 7, 8, 9 cover chin base even with slight tilt)
                chinBottomY = Math.Max((int)shape.GetPart(8).Y,
                              Math.Max((int)shape.GetPart(7).Y, (int)shape.GetPart(9).Y));

                // Eye line Y and precise horizontal center from eyes + nose bridge
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

                // Measure actual top of hair using the U2-Net silhouette mask
                hairTopY = DetectHairTopY(alphaMask, (int)face.Left, (int)face.Right, (int)face.Top, chinBottomY, faceCenterX);
            }
            finally
            {
                if (File.Exists(tempFile)) File.Delete(tempFile);
            }

            int rawHeadHeight = chinBottomY - hairTopY;
            if (rawHeadHeight <= 20)
                throw new Exception("Could not measure valid head height from facial landmarks.");

            // 3. U.S. Department of State Framing Solver:
            //    - Head height (hair top to chin): 50% - 69% of image height (target 56%)
            //    - Eye height from bottom edge: 56% - 69% of image height (target 60%)
            //    - Minimum 7% pure white headroom above hair so crown is never clipped
            float targetHeadRatio = 0.56f;
            float scale = (targetSize * targetHeadRatio) / rawHeadHeight;

            if (scale > 4.5f)
                throw new Exception("Face is too small/far away to yield a sharp 2x2\" visa photo.");

            float scaledHairTopY = hairTopY * scale;
            float scaledEyeLineY = eyeLineY * scale;
            float eyeOffsetFromHairTop = scaledEyeLineY - scaledHairTopY;

            // 60% from bottom = 40% down from top
            float desiredEyeFromTop = targetSize * (1.0f - 0.60f);
            float desiredHairTopFromTop = desiredEyeFromTop - eyeOffsetFromHairTop;

            float minHairTopFromTop = targetSize * 0.07f;
            if (desiredHairTopFromTop < minHairTopFromTop)
            {
                desiredHairTopFromTop = minHairTopFromTop;
                desiredEyeFromTop = desiredHairTopFromTop + eyeOffsetFromHairTop;

                // Keep eye height from bottom >= 56.5% (DoS minimum is 56%)
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

            // 4. Export
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

            // Anatomical guard rails relative to Dlib's brow-level face.Top:
            // Even short hair/receding hairline extends at least ~0.35x browToChin above face.Top;
            // Full hair extends up to ~0.95x browToChin above face.Top.
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

        private Bitmap GetPortraitAlphaMask(Bitmap sourceImage)
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

                            // Suppress faint curtain/background haze while preserving soft hair boundary
                            if (norm <= 0.12f)
                                norm = 0f;
                            else if (norm >= 0.88f)
                                norm = 1f;
                            else
                                norm = (norm - 0.12f) / (0.88f - 0.12f);

                            byte m = (byte)Math.Round(norm * 255f);
                            row[x * 3] = m;
                            row[x * 3 + 1] = m;
                            row[x * 3 + 2] = m;
                        }
                    }
                }
            }
            finally { smallMask.UnlockBits(maskData); }

            var fullResMask = new Bitmap(sourceImage.Width, sourceImage.Height, PixelFormat.Format24bppRgb);
            using var gm = Graphics.FromImage(fullResMask);
            gm.InterpolationMode = InterpolationMode.HighQualityBicubic;
            gm.DrawImage(smallMask, 0, 0, sourceImage.Width, sourceImage.Height);
            return fullResMask;
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
                            dRow[idx] = (byte)Math.Clamp(Math.Round(sRow[idx] * alpha + 255f * (1f - alpha)), 0, 255);
                            dRow[idx + 1] = (byte)Math.Clamp(Math.Round(sRow[idx + 1] * alpha + 255f * (1f - alpha)), 0, 255);
                            dRow[idx + 2] = (byte)Math.Clamp(Math.Round(sRow[idx + 2] * alpha + 255f * (1f - alpha)), 0, 255);
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
