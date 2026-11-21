using DlibDotNet;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.Metadata;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
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

            // U.S. Department of State digital dimensions: 600x600 min to 1200x1200 max square pixels.
            // Use 1200x1200 (max acceptable square resolution) for highest detail, and 600 DPI / 300 DPI metadata.
            int targetSize = isPrintMode ? 1200 : 600;
            double dpi = isPrintMode ? 600.0 : 300.0;

            // Head height (top of head/hair to bottom of chin): 50% - 69% of image height (target 60%).
            // Eye height from bottom of photo to eye line: 56% - 69% of image height (target 62.5%).
            const float targetHeadRatio = 0.60f;
            const float targetEyeRatioFromBottom = 0.625f;

            using var originalImage = Image.Load<Rgb24>(inputPath);
            if (originalImage.Width < 200 || originalImage.Height < 200)
                throw new Exception("Source image resolution is too low for a crisp visa photo.");

            // 1. Background matting & pure white (#FFFFFF) replacement
            using var alphaMask = GetPortraitAlphaMask(originalImage);
            using var whiteBgImage = originalImage.Clone();
            ApplyPureWhiteBackground(whiteBgImage, alphaMask);

            // 2. Detect facial landmarks on the clean white-background portrait
            string tempFile = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString() + ".bmp");
            whiteBgImage.SaveAsBmp(tempFile);

            int headTop, chinBottom, eyeLineY, faceCenterX;

            try
            {
                using var detector = Dlib.GetFrontalFaceDetector();
                using var predictor = ShapePredictor.Deserialize(_dlibModelPath);
                using var dlibImg = Dlib.LoadImage<RgbPixel>(tempFile);

                var faces = detector.Operator(dlibImg);
                if (faces.Length == 0)
                    throw new Exception("No face detected in the image.");
                if (faces.Length > 1)
                    throw new Exception("Multiple faces detected. Visa photos must contain only one person.");

                var face = faces[0];
                var shape = predictor.Detect(dlibImg, face);

                // Estimate crown/hairline slightly above the frontal face box while keeping chin at landmark 8
                int boxHeight = Math.Max(1, face.Bottom - face.Top);
                int estimatedCrown = Math.Max(0, face.Top - (int)(boxHeight * 0.14f));
                headTop = estimatedCrown;
                chinBottom = (int)shape.GetPart(8).Y;
                faceCenterX = (face.Left + face.Right) / 2;

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

            // 3. Scale so head height is ~60% of targetSize (strictly inside 50%-69%)
            float scale = (targetSize * targetHeadRatio) / currentHeadHeight;

            // Guard against severe upscaling blur
            if (scale > 4.0f)
                throw new Exception("Face in source photo is too small for high-quality visa output.");

            int scaledEyeLineY = (int)(eyeLineY * scale);
            int scaledFaceCenterX = (int)(faceCenterX * scale);

            int targetEyeLineY = targetSize - (int)(targetSize * targetEyeRatioFromBottom);
            int targetCenterX = targetSize / 2;

            int offsetX = targetCenterX - scaledFaceCenterX;
            int offsetY = targetEyeLineY - scaledEyeLineY;

            // Ensure top of head/hair has clean pure white padding if offsetY pushes top edge down
            int scaledWidth = Math.Max(1, (int)Math.Round(whiteBgImage.Width * scale));
            int scaledHeight = Math.Max(1, (int)Math.Round(whiteBgImage.Height * scale));

            using var finalImage = new Image<Rgb24>(targetSize, targetSize);
            finalImage.Mutate(ctx => ctx.BackgroundColor(Color.White));

            whiteBgImage.Mutate(ctx => ctx.Resize(new ResizeOptions
            {
                Size = new Size(scaledWidth, scaledHeight),
                Sampler = KnownResamplers.Lanczos3
            }));

            finalImage.Mutate(ctx => ctx.DrawImage(
                whiteBgImage,
                new SixLabors.ImageSharp.Point(offsetX, offsetY),
                1f));

            // 4. Embed high-DPI resolution metadata
            finalImage.Metadata.HorizontalResolution = dpi;
            finalImage.Metadata.VerticalResolution = dpi;
            finalImage.Metadata.ResolutionUnits = PixelResolutionUnit.PixelsPerInch;

            // US Visa digital uploads require JPEG (<= 240 KB); print/archival mode uses lossless PNG or 98% JPEG
            if (string.Equals(format, "PNG", StringComparison.OrdinalIgnoreCase))
            {
                finalImage.Save(outputPath, new PngEncoder
                {
                    CompressionLevel = PngCompressionLevel.BestCompression
                });
            }
            else if (isPrintMode)
            {
                finalImage.Save(outputPath, new JpegEncoder { Quality = 98 });
            }
            else
            {
                // Fit JPEG under 240 KB while preserving maximum sharpness
                SaveUnder240KbJpeg(finalImage, outputPath);
            }
        }

        private static void SaveUnder240KbJpeg(Image<Rgb24> image, string outputPath)
        {
            const int maxBytes = 240 * 1024;
            int quality = 92;
            byte[] bestBytes = Array.Empty<byte>();

            while (quality >= 50)
            {
                using var ms = new MemoryStream();
                image.Save(ms, new JpegEncoder { Quality = quality });
                byte[] bytes = ms.ToArray();
                bestBytes = bytes;
                if (bytes.Length <= maxBytes)
                    break;
                quality -= 4;
            }

            File.WriteAllBytes(outputPath, bestBytes);
        }

        private Image<L8> GetPortraitAlphaMask(Image<Rgb24> sourceImage)
        {
            const int dim = 512;
            using var resized = sourceImage.Clone(x => x.Resize(new ResizeOptions
            {
                Size = new Size(dim, dim),
                Sampler = KnownResamplers.Lanczos3
            }));

            var tensor = new DenseTensor<float>(new[] { 1, 3, dim, dim });
            resized.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < dim; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < dim; x++)
                    {
                        tensor[0, 0, y, x] = (row[x].R / 255f - 0.485f) / 0.229f;
                        tensor[0, 1, y, x] = (row[x].G / 255f - 0.456f) / 0.224f;
                        tensor[0, 2, y, x] = (row[x].B / 255f - 0.406f) / 0.225f;
                    }
                }
            });

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

            using var smallMask = new Image<L8>(dim, dim);
            smallMask.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < dim; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < dim; x++)
                    {
                        float val = outTensor[0, 0, y, x];
                        float norm = range > 1e-6f ? Math.Clamp((val - min) / range, 0f, 1f) : 0f;
                        // Clean up faint background haze while keeping soft hair edges
                        if (norm < 0.08f) norm = 0f;
                        else if (norm > 0.92f) norm = 1f;
                        row[x] = new L8((byte)Math.Round(norm * 255f));
                    }
                }
            });

            var fullResMask = smallMask.Clone(x => x.Resize(new ResizeOptions
            {
                Size = new Size(sourceImage.Width, sourceImage.Height),
                Sampler = KnownResamplers.Lanczos3
            }));

            return fullResMask;
        }

        private static void ApplyPureWhiteBackground(Image<Rgb24> image, Image<L8> mask)
        {
            image.ProcessPixelRows(mask, (imgAccessor, maskAccessor) =>
            {
                for (int y = 0; y < image.Height; y++)
                {
                    var imgRow = imgAccessor.GetRowSpan(y);
                    var maskRow = maskAccessor.GetRowSpan(y);
                    for (int x = 0; x < image.Width; x++)
                    {
                        float alpha = maskRow[x].PackedValue / 255f;
                        byte r = (byte)Math.Clamp(Math.Round(imgRow[x].R * alpha + 255f * (1f - alpha)), 0, 255);
                        byte g = (byte)Math.Clamp(Math.Round(imgRow[x].G * alpha + 255f * (1f - alpha)), 0, 255);
                        byte b = (byte)Math.Clamp(Math.Round(imgRow[x].B * alpha + 255f * (1f - alpha)), 0, 255);
                        imgRow[x] = new Rgb24(r, g, b);
                    }
                }
            });
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
