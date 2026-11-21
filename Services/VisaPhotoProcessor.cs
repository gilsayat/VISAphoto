using DlibDotNet;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.Metadata;
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace USVisaPhotoAutomator.Services
{
    public class VisaPhotoProcessor
    {
        private readonly string _onnxModelPath;
        private readonly string _dlibModelPath;

        public VisaPhotoProcessor(string modelsDirectory)
        {
            _onnxModelPath = Path.Combine(modelsDirectory, "u2net.onnx");
            _dlibModelPath = Path.Combine(modelsDirectory, "shape_predictor_68_face_landmarks.dat");
        }

        public void ProcessVisaPhoto(string inputPath, string outputPath, string format, bool isPrintMode)
        {
            if (!File.Exists(_onnxModelPath)) throw new FileNotFoundException("u2net.onnx model is missing in the Models folder.");
            if (!File.Exists(_dlibModelPath)) throw new FileNotFoundException("shape_predictor_68_face_landmarks.dat is missing.");

            int targetSize = isPrintMode ? 1200 : 600;
            float targetHeadRatio = 0.60f; 
            float targetEyeRatioFromBottom = 0.62f;

            using var originalImage = Image.Load<Rgb24>(inputPath);
            using var alphaMask = GetU2NetAlphaMask(originalImage);
            using var whiteBgImage = originalImage.Clone();
            ApplyWhiteBackground(whiteBgImage, alphaMask);

            string tempFile = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString() + ".bmp");
            whiteBgImage.SaveAsBmp(tempFile);

            int headTop, chinBottom, eyeLineY, faceCenterX;

            using (var detector = Dlib.GetFrontalFaceDetector())
            using (var predictor = ShapePredictor.Deserialize(_dlibModelPath))
            using (var dlibImg = Dlib.LoadImage<RgbPixel>(tempFile))
            {
                var faces = detector.Operator(dlibImg);
                File.Delete(tempFile);

                if (faces.Length == 0) throw new Exception("No face detected in the image.");
                if (faces.Length > 1) throw new Exception("Multiple faces detected.");

                var face = faces[0];
                var shape = predictor.Detect(dlibImg, face);

                headTop = face.Top;
                chinBottom = (int)shape.GetPart(8).Y;
                faceCenterX = (face.Left + face.Right) / 2;

                float leftEyeY = 0, rightEyeY = 0;
                for (uint i = 36; i <= 41; i++) leftEyeY += shape.GetPart(i).Y;
                for (uint i = 42; i <= 47; i++) rightEyeY += shape.GetPart(i).Y;
                
                eyeLineY = (int)((leftEyeY / 6.0f + rightEyeY / 6.0f) / 2.0f);
            }

            int currentHeadHeight = chinBottom - headTop;
            if (currentHeadHeight <= 0) throw new Exception("Invalid facial geometry detected.");

            float scale = (targetSize * targetHeadRatio) / currentHeadHeight;
            int scaledEyeLineY = (int)(eyeLineY * scale);
            int scaledFaceCenterX = (int)(faceCenterX * scale);

            int targetEyeLineY = targetSize - (int)(targetSize * targetEyeRatioFromBottom);
            int targetCenterX = targetSize / 2;
            int offsetX = targetCenterX - scaledFaceCenterX;
            int offsetY = targetEyeLineY - scaledEyeLineY;

            using var finalImage = new Image<Rgb24>(targetSize, targetSize);
            finalImage.Mutate(ctx => ctx.Fill(Color.White)); 
            
            whiteBgImage.Mutate(ctx => ctx.Resize(
                (int)(whiteBgImage.Width * scale), 
                (int)(whiteBgImage.Height * scale)));

            finalImage.Mutate(ctx => ctx.DrawImage(whiteBgImage, new Point(offsetX, offsetY), 1f));

            finalImage.Metadata.HorizontalResolution = 300;
            finalImage.Metadata.VerticalResolution = 300;
            finalImage.Metadata.ResolutionUnits = PixelResolutionUnit.PixelsPerInch;

            if (format == "PNG")
                finalImage.Save(outputPath, new PngEncoder());
            else
                finalImage.Save(outputPath, new JpegEncoder { Quality = 100 });
        }

        private Image<L8> GetU2NetAlphaMask(Image<Rgb24> sourceImage)
        {
            const int dim = 320;
            using var resized = sourceImage.Clone(x => x.Resize(dim, dim));
            var tensor = new DenseTensor<float>(new[] { 1, 3, dim, dim });
            
            resized.ProcessPixelRows(accessor => {
                for (int y = 0; y < dim; y++) {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < dim; x++) {
                        tensor[0, 0, y, x] = (row[x].R / 255f - 0.485f) / 0.229f;
                        tensor[0, 1, y, x] = (row[x].G / 255f - 0.456f) / 0.224f;
                        tensor[0, 2, y, x] = (row[x].B / 255f - 0.406f) / 0.225f;
                    }
                }
            });

            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input.1", tensor) };
            using var session = new InferenceSession(_onnxModelPath);
            using var results = session.Run(inputs);
            
            var outTensor = results.First().AsTensor<float>();
            var mask = new Image<L8>(dim, dim);
            
            float max = float.MinValue, min = float.MaxValue;
            for (int i = 0; i < dim * dim; i++) {
                float val = outTensor.GetValue(i);
                if (val > max) max = val;
                if (val < min) min = val;
            }

            mask.ProcessPixelRows(accessor => {
                for (int y = 0; y < dim; y++) {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < dim; x++) {
                        float val = outTensor[0, 0, y, x];
                        float norm = (val - min) / (max - min);
                        row[x] = new L8((byte)(norm * 255));
                    }
                }
            });

            mask.Mutate(x => x.Resize(sourceImage.Width, sourceImage.Height));
            return mask;
        }

        private void ApplyWhiteBackground(Image<Rgb24> image, Image<L8> mask)
        {
            image.ProcessPixelRows(mask, (imgAccessor, maskAccessor) => {
                for (int y = 0; y < image.Height; y++) {
                    var imgRow = imgAccessor.GetRowSpan(y);
                    var maskRow = maskAccessor.GetRowSpan(y);
                    for (int x = 0; x < image.Width; x++) {
                        float alpha = maskRow[x].PackedValue / 255f;
                        byte r = (byte)(imgRow[x].R * alpha + 255 * (1 - alpha));
                        byte g = (byte)(imgRow[x].G * alpha + 255 * (1 - alpha));
                        byte b = (byte)(imgRow[x].B * alpha + 255 * (1 - alpha));
                        imgRow[x] = new Rgb24(r, g, b);
                    }
                }
            });
        }
    }
}