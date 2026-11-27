using DlibDotNet;
using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;

namespace USVisaPhotoAutomator.Services
{
    public enum PrintLayout { Digital2x2, Print4x6, Print4x8, Print5x10 }
    public enum BackgroundColorOption { White, OffWhite }

    public class ProcessingRequest
    {
        public string OutputPath { get; set; } = "";
        public PrintLayout Layout { get; set; }
        public BackgroundColorOption BgColor { get; set; }
        public double Tilt { get; set; }
        public double ZoomOffset { get; set; }
        public double PanX { get; set; }
        public double PanY { get; set; }
    }

    public struct ComplianceCheckItem { public string Label { get; set; } public string ValueText { get; set; } public bool Pass { get; set; } }
    public struct ComplianceReport { public ComplianceCheckItem HeadHeight { get; set; } public ComplianceCheckItem EyeLevel { get; set; } public ComplianceCheckItem Centering { get; set; } public ComplianceCheckItem HeadTilt { get; set; } public bool OverallPass { get; set; } }

    public sealed class VisaPhotoProcessor : IDisposable
    {
        private readonly string _dlibModelPath;
        private readonly object _modelLock = new object();
        private FrontalFaceDetector? _detector;
        private ShapePredictor? _predictor;
        private bool _isDisposed;

        private Bitmap? _masterOriginal;
        private Bitmap? _masterExtracted;
        
        private float _masterEyeX, _masterEyeY, _masterHairTopY, _masterChinY;
        private float _detectedEyeSlopeDeg = 0;
        private float _autoTiltDeg = 0;

        public VisaPhotoProcessor()
        {
            _dlibModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "shape_predictor_68_face_landmarks.dat");
        }

        public void PreWarmModels()
        {
            if (_isDisposed) return;
            lock (_modelLock)
            {
                if (_detector != null) return;
                if (!File.Exists(_dlibModelPath)) return;
                _detector = Dlib.GetFrontalFaceDetector();
                _predictor = ShapePredictor.Deserialize(_dlibModelPath);
            }
        }

        public void InitializeMasterFile(string originalPath, byte[] extractedPngBytes)
        {
            PreWarmModels();
            if (_detector == null || _predictor == null) throw new Exception("Dlib Models missing.");

            _masterOriginal?.Dispose();
            _masterExtracted?.Dispose();

            _masterOriginal = LoadOrientedBitmap(originalPath);
            
            // Load the perfectly matted PNG provided by the WebGPU JS Engine
            using (var ms = new MemoryStream(extractedPngBytes))
            {
                _masterExtracted = new Bitmap(ms);
            }

            using Bitmap proxy = ResizeIfLarger(_masterOriginal, 800);
            float ratio = (float)_masterOriginal.Width / proxy.Width;
            
            bool faceFound = false;
            string tempFile = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".bmp");
            proxy.Save(tempFile, ImageFormat.Bmp);
            try
            {
                using var dlibImg = Dlib.LoadImage<RgbPixel>(tempFile);
                var faces = _detector.Operator(dlibImg, 0);
                if (faces.Length == 0) faces = _detector.Operator(dlibImg, 1);
                
                if (faces.Length > 0)
                {
                    faceFound = true;
                    var face = faces.OrderByDescending(r => r.Width * r.Height).First();
                    using var shape = _predictor.Detect(dlibImg, face);

                    float lY = 0, rY = 0, lX = 0, rX = 0;
                    for (uint i=36; i<=41; i++) { lX += shape.GetPart(i).X; lY += shape.GetPart(i).Y; }
                    for (uint i=42; i<=47; i++) { rX += shape.GetPart(i).X; rY += shape.GetPart(i).Y; }
                    
                    _masterEyeX = ((((lX/6f + rX/6f)/2f) + shape.GetPart(27).X)/2f) * ratio;
                    _masterEyeY = ((lY/6f + rY/6f)/2f) * ratio;
                    _masterChinY = (Math.Max((int)shape.GetPart(8).Y, (int)shape.GetPart(9).Y) * ratio);
                    
                    float avgLeftX = (lX/6f)*ratio; float avgLeftY = (lY/6f)*ratio;
                    float avgRightX = (rX/6f)*ratio; float avgRightY = (rY/6f)*ratio;
                    _detectedEyeSlopeDeg = (float)(Math.Atan2(avgRightY - avgLeftY, avgRightX - avgLeftX) * 180 / Math.PI);
                    _autoTiltDeg = Math.Clamp(-_detectedEyeSlopeDeg, -15f, 15f);
                }
            }
            finally { if (File.Exists(tempFile)) File.Delete(tempFile); }

            if (!faceFound)
            {
                _masterEyeX = _masterOriginal.Width / 2f;
                _masterEyeY = _masterOriginal.Height * 0.4f;
                _masterChinY = _masterOriginal.Height * 0.65f;
                _detectedEyeSlopeDeg = 0;
                _autoTiltDeg = 0;
            }

            // Find top of hair from the extracted alpha
            _masterHairTopY = DetectHairTop(_masterExtracted, _masterEyeX, _masterEyeY, _masterChinY);
        }

        private (float scale, float targetEyeYOnCanvas, float panXPx, float panYPx, float totalTiltDeg) GetTransformParams(ProcessingRequest req, int targetSize)
        {
            float origH = _masterOriginal != null ? _masterOriginal.Height : targetSize;
            int rawHeadHeight = (int)(_masterChinY - _masterHairTopY);
            if (rawHeadHeight <= 20) rawHeadHeight = (int)(origH * 0.5f);

            float baseScale = (targetSize * 0.58f) / rawHeadHeight;
            float scale = baseScale * (1.0f + (float)req.ZoomOffset / 100f);
            
            float targetEyeYOnCanvas = targetSize * 0.375f; 
            float panXPx = (float)req.PanX * (targetSize * 0.01f); 
            float panYPx = (float)req.PanY * (targetSize * 0.01f);
            float totalTiltDeg = _autoTiltDeg + (float)req.Tilt;

            return (scale, targetEyeYOnCanvas, panXPx, panYPx, totalTiltDeg);
        }

        public ComplianceReport ComputeCompliance(ProcessingRequest req)
        {
            int targetSize = 600; 
            var t = GetTransformParams(req, targetSize);
            float rad = (float)(t.totalTiltDeg * Math.PI / 180.0);
            float cos = (float)Math.Cos(rad);
            float sin = (float)Math.Sin(rad);

            PointF TransformPoint(float x, float y) {
                float dx = (x - _masterEyeX) * t.scale;
                float dy = (y - _masterEyeY) * t.scale;
                float rx = dx * cos - dy * sin;
                float ry = dx * sin + dy * cos;
                return new PointF((targetSize / 2f) + t.panXPx + rx, t.targetEyeYOnCanvas + t.panYPx + ry);
            }

            PointF eyePt = TransformPoint(_masterEyeX, _masterEyeY);
            PointF hairPt = TransformPoint(_masterEyeX, _masterHairTopY);
            PointF chinPt = TransformPoint(_masterEyeX, _masterChinY);

            float headHeightFrac = Math.Max(0, (chinPt.Y - hairPt.Y) / targetSize);
            float headHeightPct = headHeightFrac * 100f;
            float headHeightIn = headHeightFrac * 2.0f;
            bool headPass = headHeightPct >= 50.0f && headHeightPct <= 69.0f;

            float eyeFromBottomFrac = 1.0f - (eyePt.Y / targetSize);
            float eyeFromBottomPct = eyeFromBottomFrac * 100f;
            float eyeFromBottomIn = eyeFromBottomFrac * 2.0f;
            bool eyePass = eyeFromBottomPct >= 56.0f && eyeFromBottomPct <= 69.0f;

            float centerOffsetPct = ((eyePt.X / targetSize) - 0.5f) * 100f;
            bool centerPass = Math.Abs(centerOffsetPct) <= 3.5f;
            float effectiveTiltDeg = _detectedEyeSlopeDeg + t.totalTiltDeg;
            bool tiltPass = Math.Abs(effectiveTiltDeg) <= 2.0f;

            return new ComplianceReport {
                HeadHeight = new ComplianceCheckItem { Label = "Head Height", ValueText = $"{headHeightPct:F1}% ({headHeightIn:F2} in)", Pass = headPass },
                EyeLevel = new ComplianceCheckItem { Label = "Eye Level", ValueText = $"{eyeFromBottomPct:F1}% ({eyeFromBottomIn:F2} in)", Pass = eyePass },
                Centering = new ComplianceCheckItem { Label = "Centering", ValueText = $"{centerOffsetPct:F1}%", Pass = centerPass },
                HeadTilt = new ComplianceCheckItem { Label = "Head Level", ValueText = $"{effectiveTiltDeg:F1}°", Pass = tiltPass },
                OverallPass = headPass && eyePass && centerPass && tiltPass
            };
        }

        public byte[] RenderFrame(ProcessingRequest req, bool previewMode)
        {
            if (_masterExtracted == null) throw new Exception("Engine not initialized.");

            int targetSize = previewMode ? 600 : 1200; 
            byte bgTarget = req.BgColor == BackgroundColorOption.OffWhite ? (byte)242 : (byte)255;

            using Bitmap canvas = new Bitmap(targetSize, targetSize, PixelFormat.Format32bppArgb);
            using Graphics g = Graphics.FromImage(canvas);
            
            if (!previewMode) g.Clear(System.Drawing.Color.FromArgb(bgTarget, bgTarget, bgTarget));
            else g.Clear(System.Drawing.Color.Transparent);

            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = PixelOffsetMode.Half; 
            g.SmoothingMode = SmoothingMode.HighQuality;

            var t = GetTransformParams(req, targetSize);

            g.TranslateTransform((targetSize / 2f) + t.panXPx, t.targetEyeYOnCanvas + t.panYPx);
            g.RotateTransform(t.totalTiltDeg);
            g.ScaleTransform(t.scale, t.scale);
            g.TranslateTransform(-_masterEyeX, -_masterEyeY);
            
            g.DrawImage(_masterExtracted, 0, 0);
            
            g.ResetTransform();

            using var ms = new MemoryStream();
            if (previewMode) {
                canvas.Save(ms, ImageFormat.Png); 
            } else {
                var ep = new EncoderParameters(1);
                ep.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, 98L); 
                var codec = ImageCodecInfo.GetImageEncoders().First(c => c.FormatID == ImageFormat.Jpeg.Guid);
                canvas.Save(ms, codec, ep);
            }
            return ms.ToArray();
        }

        public void GenerateHighResPrint(ProcessingRequest req)
        {
            if (_masterOriginal == null) return;
            
            byte[] baseBytes = RenderFrame(req, previewMode: false);
            using var ms = new MemoryStream(baseBytes);
            using Bitmap base2x2 = new Bitmap(ms);
            base2x2.SetResolution(600, 600);

            using Bitmap finalSheet = (req.Layout == PrintLayout.Digital2x2) ? new Bitmap(base2x2) : GenerateSheet(base2x2, req.Layout, 600, req.BgColor);
            SaveInjectingEXIF(finalSheet, req.OutputPath, _masterOriginal, req.Layout == PrintLayout.Digital2x2 ? 90 : 98, 600);
        }

        private static int DetectHairTop(Bitmap mask, float fcx, float fcy, float chinY)
        {
            int scanL = Math.Clamp((int)(fcx - 50), 0, mask.Width - 1);
            int scanR = Math.Clamp((int)(fcx + 50), scanL, mask.Width - 1);
            int startY = (int)fcy;
            int yLim = Math.Max(0, (int)(fcy - (chinY - fcy)));
            
            var md = mask.LockBits(new System.Drawing.Rectangle(0,0,mask.Width,mask.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            int top = startY;
            unsafe {
                byte* ptr = (byte*)md.Scan0;
                for (int y = startY; y >= yLim; y--) {
                    bool hair = false;
                    for (int x = scanL; x <= scanR; x++) if (ptr[(y*md.Stride)+(x*4)+3] > 100) { hair = true; break; }
                    if (hair) top = y;
                }
            }
            mask.UnlockBits(md);
            return top;
        }

        private static Bitmap GenerateSheet(Bitmap base2x2, PrintLayout layout, int dpi, BackgroundColorOption bg)
        {
            int t = base2x2.Width;
            int gutter = 30; 
            
            int w = 4*dpi, h = 6*dpi, cols = 2, rows = 3;
            int sX = (w - (cols*t + (cols-1)*gutter))/2;
            int sY = (h - (rows*t + (rows-1)*gutter))/2;

            if (layout == PrintLayout.Print4x8) { h = 8*dpi; rows = 4; sY = (h - (rows*t + (rows-1)*gutter))/2; }
            if (layout == PrintLayout.Print5x10) { w = 5*dpi; h = 10*dpi; rows = 4; sX = (w - (cols*t + (cols-1)*gutter))/2; sY = (h - (rows*t + (rows-1)*gutter))/2; }
            
            var sheet = new System.Drawing.Bitmap(w, h, PixelFormat.Format24bppRgb);
            sheet.SetResolution(dpi, dpi);
            using Graphics gs = Graphics.FromImage(sheet);
            gs.Clear(System.Drawing.Color.White);
            
            for (int y = 0; y < rows; y++) {
                for (int x = 0; x < cols; x++) {
                    int pX = sX + x * (t + gutter);
                    int pY = sY + y * (t + gutter);
                    gs.DrawImageUnscaled(base2x2, pX, pY);
                    gs.DrawRectangle(new Pen(System.Drawing.Color.Black, 2f), pX, pY, t, t);
                }
            }
            return sheet;
        }

        private static void SaveInjectingEXIF(Bitmap finalSheet, string path, Bitmap orig, int quality, int dpi)
        {
            var codec = ImageCodecInfo.GetImageEncoders().FirstOrDefault(c => c.FormatID == ImageFormat.Jpeg.Guid);
            if (codec == null) { finalSheet.Save(path, ImageFormat.Jpeg); return; }

            if (Array.IndexOf(orig.PropertyIdList, 0x8773) >= 0) {
                var icc = orig.GetPropertyItem(0x8773);
                if (icc != null) finalSheet.SetPropertyItem(icc);
            }

            using var ep = new EncoderParameters(1);
            ep.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, (long)quality);
            finalSheet.Save(path, codec, ep);
        }

        private static Bitmap LoadOrientedBitmap(string path) {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var raw = System.Drawing.Image.FromStream(fs, false, false);
            var b = new Bitmap(raw);
            if (Array.IndexOf(raw.PropertyIdList, 0x0112) >= 0) {
                var p = raw.GetPropertyItem(0x0112);
                if (p?.Value != null) switch (BitConverter.ToUInt16(p.Value, 0)) {
                    case 3: b.RotateFlip(RotateFlipType.Rotate180FlipNone); break;
                    case 6: b.RotateFlip(RotateFlipType.Rotate90FlipNone); break;
                    case 8: b.RotateFlip(RotateFlipType.Rotate270FlipNone); break;
                }
            }
            return b;
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

        public void Dispose() 
        { 
            lock(_modelLock) { _detector?.Dispose(); _predictor?.Dispose(); _isDisposed = true; } 
            _masterOriginal?.Dispose(); _masterExtracted?.Dispose();
        }
    }
}
