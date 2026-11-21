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
        private readonly object _sessionLock = new object();
        private InferenceSession? _session;

        public VisaPhotoProcessor(string modelsDirectory)
        {
            _onnxModelPath = Path.Combine(modelsDirectory, "u2net.onnx");
            _dlibModelPath = Path.Combine(modelsDirectory, "shape_predictor_68_face_landmarks.dat");
        }

        private InferenceSession GetOrCreateSession()
        {
            if (_session != null) return _session;
            lock (_sessionLock)
            {
                if (_session != null) return _session;
                if (!File.Exists(_onnxModelPath))
                    throw new FileNotFoundException("u2net.onnx model is missing in the Models folder.");

                var opts = new SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
                };
                _session = new InferenceSession(_onnxModelPath, opts);
                return _session;
            }
        }

        public void ProcessVisaPhoto(string inputPath, string outputPath, string format, bool isPrintMode)
        {
            if (!File.Exists(_onnxModelPath))
                throw new FileNotFoundException("u2net.onnx model is missing in the Models folder.");
            if (!File.Exists(_dlibModelPath))
                throw new FileNotFoundException("shape_predictor_68_face_landmarks.dat is missing.");

            // U.S. Department of State digital square size: 600x600 min to 1200x1200 max pixels.
            int targetSize = isPrintMode ? 1200 : 600;
            float dpi = isPrintMode ? 600f : 300f;

            // Head height (top of head/hair to chin): 50% - 69% of image height (target 60%).
            // Eye height from bottom of photo to eye line: 56% - 69% of image height (target 62.5%).
            const float targetHeadRatio = 0.60f;
            const float targetEyeRatioFromBottom = 0.625f;

            using Bitmap originalImage = LoadOrientedBitmap(inputPath);
            if (originalImage.Width < 200 || originalImage.Height < 200)
                throw new Exception("Source image resolution is too low for a crisp visa photo.");

            // Cap working resolution at 1600px on longest side for reliable detection & speed
            using Bitmap workingImage = ResizeIfLarger(originalImage, 1600);

            // 1. Background matting & pure white (#FFFFFF) replacement
            using Bitmap alphaMask = GetPortraitAlphaMask(workingImage);
            using Bitmap whiteBgImage = ApplyPureWhiteBackground(workingImage, alphaMask);

            // 2. Detect facial landmarks on the upright, white-background portrait
            string tempFile = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".bmp");
            whiteBgImage.Save(tempFile, ImageFormat.Bmp);

            int headTop, chinBottom, eyeLineY, faceCenterX;

            try
            {
                using var detector = Dlib.GetFrontalFaceDetector();
                using var predictor = ShapePredictor.Deserialize(_dlibModelPath);
                using var dlibImg = Dlib.LoadImage<RgbPixel>(tempFile);

                var faces = detector.Operator(dlibImg, 1);
                if (faces.Length == 0)
                    throw new Exception("No face detected in the image. Ensure the face is forward-facing and upright.");
                if (faces.Length > 1)
                    throw new Exception("Multiple faces detected. Visa photos must contain only one person.");

                var face = faces.OrderByDescending(r => r.Width * r.Height).First();
                using var shape = predictor.Detect(dlibImg, face);

                int boxHeight = Math.Max(1, (int)face.Height);
                int estimatedCrown = Math.Max(0, (int)face.Top - (int)(boxHeight * 0.14f));
                headTop = estimatedCrown;
                chinBottom = (int)shape.GetPart(8).Y;
                faceCenterX = ((int)face.Left + (int)face.Right) / 2;

                float leftEyeY = 0, rightEyeY = 0;
                for (uint i = 36; i <= 41; i++) leftEyeY += shape.GetPart(i).Y;
                for (uint i = 42; i <= 47; i++) rightEyeY += shape.GetPart(i).Y;
                eyeLineY = (int)((leftEyeY / 6.0f + rightEyeY / 6.0f) / 2.0f);
            }
            finally
            {
                if (File.Exists(tempFile))
                    File.Delete(tempFile);
            }

            int currentHeadHeight = chinBottom - headTop;
            if (currentHeadHeight <= 0)
                throw new Exception("Could not measure valid head height.");

            // 3. Scale so head height is ~60% of targetSize (inside 50% - 69%)
            float scale = (targetSize * targetHeadRatio) / currentHeadHeight;
            if (scale > 4.0f)
                throw new Exception("Face in source photo is too small for high-quality visa output.");

            int scaledEyeLineY = (int)(eyeLineY * scale);
            int scaledFaceCenterX = (int)(faceCenterX * scale);

            int targetEyeLineY = targetSize - (int)(targetSize * targetEyeRatioFromBottom);
            int targetCenterX = targetSize / 2;

            int offsetX = targetCenterX - scaledFaceCenterX;
            int offsetY = targetEyeLineY - scaledEyeLineY;

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

            // 4. Export format & size rules
            if (string.Equals(format, "PNG", StringComparison.OrdinalIgnoreCase))
            {
                finalImage.Save(outputPath, ImageFormat.Png);
            }
            else if (isPrintMode)
            {
                SaveJpegWithQuality(finalImage, outputPath, 98);
            }
            else
            {
                SaveUnder240KbJpeg(finalImage, outputPath);
            }
        }

        private static Bitmap LoadOrientedBitmap(string inputPath)
        {
            using var rawImg = System.Drawing.Image.FromFile(inputPath);
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
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;
            g.SmoothingMode = SmoothingMode.HighQuality;
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
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.DrawImage(sourceImage, 0, 0, dim, dim);
            }

            var tensor = new DenseTensor<float>(new[] { 1, 3, dim, dim });
            var rect = new Rectangle(0, 0, dim, dim);
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
                            // Format24bppRgb in System.Drawing is B, G, R in memory
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
            finally
            {
                resized.UnlockBits(bmpData);
            }

            var session = GetOrCreateSession();
            string inputName = session.InputMetadata.Keys.FirstOrDefault() ?? "input.1";
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensor) };

            using var results = session.Run(inputs);
            var outTensor = results.First().AsTensor<float>();

            float min = float.MaxValue, max = float.MinValue;
            foreach (float val in outTensor)
            {
                if (val < min) min = val;
                if (val > max) max = val;
            }
            float range = max - min;

            using var smallMask = new Bitmap(dim, dim, PixelFormat.Format24bppRgb);
            var maskRect = new Rectangle(0, 0, dim, dim);
            var maskData = smallMask.LockBits(maskRect, ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);

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
                            if (norm < 0.08f) norm = 0f;
                            else if (norm > 0.92f) norm = 1f;

                            byte m = (byte)Math.Round(norm * 255f);
                            row[x * 3] = m;
                            row[x * 3 + 1] = m;
                            row[x * 3 + 2] = m;
                        }
                    }
                }
            }
            finally
            {
                smallMask.UnlockBits(maskData);
            }

            var fullResMask = new Bitmap(sourceImage.Width, sourceImage.Height, PixelFormat.Format24bppRgb);
            using var gm = Graphics.FromImage(fullResMask);
            gm.InterpolationMode = InterpolationMode.HighQualityBicubic;
            gm.PixelOffsetMode = PixelOffsetMode.HighQuality;
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

            var rect = new Rectangle(0, 0, width, height);
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

                            byte b = (byte)Math.Clamp(Math.Round(sRow[idx] * alpha + 255f * (1f - alpha)), 0, 255);
                            byte g = (byte)Math.Clamp(Math.Round(sRow[idx + 1] * alpha + 255f * (1f - alpha)), 0, 255);
                            byte r = (byte)Math.Clamp(Math.Round(sRow[idx + 2] * alpha + 255f * (1f - alpha)), 0, 255);

                            dRow[idx] = b;
                            dRow[idx + 1] = g;
                            dRow[idx + 2] = r;
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

        private static ImageCodecInfo? GetJpegCodec()
        {
            return ImageCodecInfo.GetImageEncoders().FirstOrDefault(c => c.FormatID == ImageFormat.Jpeg.Guid);
        }

        private static void SaveJpegWithQuality(Bitmap image, string outputPath, int quality)
        {
            var codec = GetJpegCodec();
            if (codec == null)
            {
                image.Save(outputPath, ImageFormat.Jpeg);
                return;
            }

            using var encoderParams = new EncoderParameters(1);
            encoderParams.Param[0] = new EncoderParameter(Encoder.Quality, (long)quality);
            image.Save(outputPath, codec, encoderParams);
        }

        private static void SaveUnder240KbJpeg(Bitmap image, string outputPath)
        {
            const int maxBytes = 240 * 1024;
            var codec = GetJpegCodec();
            if (codec == null)
            {
                image.Save(outputPath, ImageFormat.Jpeg);
                return;
            }

            int quality = 92;
            byte[] bestBytes = Array.Empty<byte>();

            while (quality >= 50)
            {
                using var ms = new MemoryStream();
                using var encoderParams = new EncoderParameters(1);
                encoderParams.Param[0] = new EncoderParameter(Encoder.Quality, (long)quality);
                image.Save(ms, codec, encoderParams);

                bestBytes = ms.ToArray();
                if (bestBytes.Length <= maxBytes)
                    break;
                quality -= 4;
            }

            File.WriteAllBytes(outputPath, bestBytes);
        }

        public void Dispose()
        {
            lock (_sessionLock)
            {
                _session?.Dispose();
                _session = null;
            }
        }
    }
}
