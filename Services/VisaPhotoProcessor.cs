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

    public class ProcessingRequest
    {
        public string OutputPath { get; set; } = "";
        public PrintLayout Layout { get; set; }
        public BackgroundColorOption BgColor { get; set; }
        public EdgeFeatherMode EdgeMode { get; set; }
        public double Tilt { get; set; }
        public double ZoomOffset { get; set; }
        public double PanX { get; set; }
        public double PanY { get; set; }
        public double Brightness { get; set; }
        public double HighlightRecovery { get; set; }
        public double ColorTemp { get; set; }
        public int EdgeChoke { get; set; }
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

        // Master State (Held in Memory for Live Real-Time Rendering)
        private Bitmap? _masterOriginal;
        private Bitmap? _masterAlpha;
        private float _masterEyeX, _masterEyeY, _masterHairTopY, _masterChinY;
        private bool _faceFound;

        public VisaPhotoProcessor(string modelsDirectory)
        {
            _onnxModelPath = Path.Combine(modelsDirectory, "u2net.onnx");
            _dlibModelPath = Path.Combine(modelsDirectory, "shape_predictor_68_face_landmarks.dat");
        }

        public void PreWarmModels()
        {
            if (_isDisposed) return;
            lock (_modelLock)
            {
                if (_session != null) return;
                if (!File.Exists(_onnxModelPath) || !File.Exists(_dlibModelPath)) return;
                _session = new InferenceSession(_onnxModelPath, new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL });
                _detector = Dlib.GetFrontalFaceDetector();
                _predictor = ShapePredictor.Deserialize(_dlibModelPath);
            }
        }

        public void InitializeMasterFile(string filePath)
        {
            PreWarmModels();
            if (_session == null || _detector == null || _predictor == null) throw new Exception("AI Models missing.");

            _masterOriginal?.Dispose();
            _masterAlpha?.Dispose();

            // Load and orient
            _masterOriginal = LoadOrientedBitmap(filePath);

            // Proxy Landmark Detection (Fast)
            using Bitmap proxy = ResizeIfLarger(_masterOriginal, 800);
            float ratio = (float)_masterOriginal.Width / proxy.Width;
            
            _faceFound = false;
            string tempFile = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".bmp");
            proxy.Save(tempFile, ImageFormat.Bmp);
            try
            {
                using var dlibImg = Dlib.LoadImage<RgbPixel>(tempFile);
                var faces = _detector.Operator(dlibImg, 0);
                if (faces.Length == 0) faces = _detector.Operator(dlibImg, 1);
                
                if (faces.Length > 0)
                {
                    _faceFound = true;
                    var face = faces.OrderByDescending(r => r.Width * r.Height).First();
                    using var shape = _predictor.Detect(dlibImg, face);

                    float lY = 0, rY = 0, lX = 0, rX = 0;
                    for (uint i=36; i<=41; i++) { lX += shape.GetPart(i).X; lY += shape.GetPart(i).Y; }
                    for (uint i=42; i<=47; i++) { rX += shape.GetPart(i).X; rY += shape.GetPart(i).Y; }
                    
                    _masterEyeX = ((((lX/6f + rX/6f)/2f) + shape.GetPart(27).X)/2f) * ratio;
                    _masterEyeY = ((lY/6f + rY/6f)/2f) * ratio;
                    _masterChinY = (Math.Max((int)shape.GetPart(8).Y, (int)shape.GetPart(9).Y) * ratio);
                }
            }
            finally { if (File.Exists(tempFile)) File.Delete(tempFile); }

            if (!_faceFound)
            {
                _masterEyeX = _masterOriginal.Width / 2f;
                _masterEyeY = _masterOriginal.Height * 0.4f;
                _masterChinY = _masterOriginal.Height * 0.65f;
            }

            // Extract Alpha Mask
            _masterAlpha = GetCoarseMask(_masterOriginal);
            
            // Find Top of Hair
            _masterHairTopY = _faceFound ? DetectHairTop(_masterAlpha, _masterEyeX, _masterEyeY, _masterChinY) : (_masterOriginal.Height * 0.2f);
        }

        // Live Real-Time Engine (Runs in < 30ms when dragging slider)
        public byte[] RenderFrame(ProcessingRequest req, bool previewMode)
        {
            if (_masterOriginal == null || _masterAlpha == null) throw new Exception("Master not initialized.");

            int targetSize = previewMode ? 600 : 1200; // Fast 600x600 for UI, 1200x1200 (600 DPI) for final
            byte bgTarget = req.BgColor == BackgroundColorOption.OffWhite ? (byte)242 : (byte)255;

            // 1. Create Sub-Pixel Color & Choke Matrix
            using Bitmap colorMaskedSubject = ProcessSubjectLayer(_masterOriginal, _masterAlpha, req);

            // 2. Set up Canvas
            using Bitmap canvas = new Bitmap(targetSize, targetSize, PixelFormat.Format24bppRgb);
            using Graphics g = Graphics.FromImage(canvas);
            g.Clear(System.Drawing.Color.FromArgb(bgTarget, bgTarget, bgTarget));
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;
            g.SmoothingMode = SmoothingMode.HighQuality;

            // 3. Compute Scale & Transforms
            int rawHeadHeight = (int)(_masterChinY - _masterHairTopY);
            if (rawHeadHeight <= 20) rawHeadHeight = (int)(_masterOriginal.Height * 0.5f);

            // Visa rule: Head must be 50-69%. We anchor base at 56%.
            float baseScale = (targetSize * 0.56f) / rawHeadHeight;
            float scale = baseScale * (1.0f + (float)req.ZoomOffset / 100f);

            // Eyes must be 56-69% from bottom. Target = 60% from bottom (40% from top)
            float targetEyeYOnCanvas = targetSize * 0.40f; 
            
            float panX = (float)req.PanX * (targetSize * 0.01f); // Absolute percentage panning
            float panY = (float)req.PanY * (targetSize * 0.01f);

            // The Matrix Magic: Rotate and Scale around the EYE coordinate, then move to Canvas Safe Zone.
            g.TranslateTransform((targetSize / 2f) + panX, targetEyeYOnCanvas + panY);
            g.RotateTransform((float)req.Tilt);
            g.ScaleTransform(scale, scale);
            g.TranslateTransform(-_masterEyeX, -_masterEyeY);
            
            // Draw the composited subject exactly into position
            g.DrawImage(colorMaskedSubject, 0, 0);

            // Print Borders (Only if not single digital)
            if (!previewMode && req.Layout != PrintLayout.Digital2x2)
            {
                g.ResetTransform();
                g.DrawRectangle(new Pen(System.Drawing.Color.Black, 2f), 4, 4, targetSize - 8, targetSize - 8);
            }

            using var ms = new MemoryStream();
            var ep = new EncoderParameters(1);
            ep.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, previewMode ? 70L : 95L); // Fast compression for UI
            var codec = ImageCodecInfo.GetImageEncoders().First(c => c.FormatID == ImageFormat.Jpeg.Guid);
            canvas.Save(ms, codec, ep);
            return ms.ToArray();
        }

        public void GenerateHighResPrint(ProcessingRequest req)
        {
            if (_masterOriginal == null) return;
            
            // Render the base 1200x1200 2x2
            byte[] baseBytes = RenderFrame(req, previewMode: false);
            
            using var ms = new MemoryStream(baseBytes);
            using Bitmap base2x2 = new Bitmap(ms);
            base2x2.SetResolution(600, 600);

            // Print Layout Generator
            using Bitmap finalSheet = (req.Layout == PrintLayout.Digital2x2) ? new Bitmap(base2x2) : GenerateSheet(base2x2, req.Layout, 600, req.BgColor);

            // Save with EXIF / ICC Injection
            SaveInjectingEXIF(finalSheet, req.OutputPath, _masterOriginal, req.Layout == PrintLayout.Digital2x2 ? 90 : 98, 600);
        }

        // Extremely fast pointer-based loop to combine Color Grading and Alpha Masking directly into a 32-bit ARGB layer
        private static Bitmap ProcessSubjectLayer(Bitmap orig, Bitmap alpha, ProcessingRequest req)
        {
            int w = orig.Width, h = orig.Height;
            Bitmap argb = new Bitmap(w, h, PixelFormat.Format32bppArgb);
            
            var od = orig.LockBits(new System.Drawing.Rectangle(0,0,w,h), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            var ad = alpha.LockBits(new System.Drawing.Rectangle(0,0,w,h), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            var rd = argb.LockBits(new System.Drawing.Rectangle(0,0,w,h), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);

            // Gamma LUT for Brightness & Highlight Recovery
            float bMod = (float)(req.Brightness / 100f);
            float rMod = (float)(req.HighlightRecovery / 100f);
            float tempModR = 1.0f + ((float)req.ColorTemp / 150f);
            float tempModB = 1.0f - ((float)req.ColorTemp / 150f);

            byte[] lut = new byte[256];
            for(int i=0; i<256; i++) {
                float norm = i/255f;
                float lifted = norm + (bMod * (1f - norm) * norm * 4f); 
                if (lifted > 0.8f && rMod > 0) lifted = lifted - (rMod * (lifted - 0.8f));
                lut[i] = (byte)Math.Clamp(lifted * 255f, 0, 255);
            }

            float chokeOffset = (float)req.EdgeChoke * 0.05f;

            unsafe {
                byte* oPtr = (byte*)od.Scan0; byte* aPtr = (byte*)ad.Scan0; byte* rPtr = (byte*)rd.Scan0;
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int idx24 = (y * od.Stride) + (x * 3);
                        int idx32 = (y * rd.Stride) + (x * 4);

                        // Colors
                        rPtr[idx32] = (byte)Math.Clamp(lut[oPtr[idx24]] * tempModB, 0, 255);       // B
                        rPtr[idx32+1] = lut[oPtr[idx24+1]];                                        // G
                        rPtr[idx32+2] = (byte)Math.Clamp(lut[oPtr[idx24+2]] * tempModR, 0, 255);   // R

                        // Mask Choke
                        float val = aPtr[idx24]/255f;
                        val = Math.Clamp(val + chokeOffset, 0f, 1f);
                        
                        float a = 0f;
                        if (val > 0.8f) a = 1f;
                        else if (val > 0.2f) { float t = (val - 0.2f)/0.6f; a = t*t*(3-2*t); }
                        
                        rPtr[idx32+3] = (byte)(a * 255f); // Alpha
                    }
                }
            }
            orig.UnlockBits(od); alpha.UnlockBits(ad); argb.UnlockBits(rd);
            return argb;
        }

        private Bitmap GetCoarseMask(Bitmap source)
        {
            using var resized = new Bitmap(320, 320, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(resized)) { g.InterpolationMode = InterpolationMode.HighQualityBicubic; g.DrawImage(source, 0, 0, 320, 320); }
            var tensor = new DenseTensor<float>(new[] { 1, 3, 320, 320 });
            var bd = resized.LockBits(new System.Drawing.Rectangle(0, 0, 320, 320), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            unsafe
            {
                byte* ptr = (byte*)bd.Scan0;
                for (int y = 0; y < 320; y++) for (int x = 0; x < 320; x++) {
                    tensor[0,0,y,x] = (ptr[(y*bd.Stride)+(x*3)+2]/255f - 0.485f)/0.229f;
                    tensor[0,1,y,x] = (ptr[(y*bd.Stride)+(x*3)+1]/255f - 0.456f)/0.224f;
                    tensor[0,2,y,x] = (ptr[(y*bd.Stride)+(x*3)]/255f - 0.406f)/0.225f;
                }
            }
            resized.UnlockBits(bd);
            using var res = _session!.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_session.InputMetadata.Keys.First(), tensor) });
            var ot = res.First().AsTensor<float>();

            using var sm = new Bitmap(320, 320, PixelFormat.Format24bppRgb);
            var sd = sm.LockBits(new System.Drawing.Rectangle(0,0,320,320), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
            unsafe {
                byte* ptr = (byte*)sd.Scan0;
                for (int y = 0; y < 320; y++) for (int x = 0; x < 320; x++) {
                    byte m = (byte)(Math.Clamp(ot[0,0,y,x], 0f, 1f) * 255f);
                    ptr[(y*sd.Stride)+(x*3)] = m; ptr[(y*sd.Stride)+(x*3)+1] = m; ptr[(y*sd.Stride)+(x*3)+2] = m;
                }
            }
            sm.UnlockBits(sd);
            var output = new Bitmap(source.Width, source.Height, PixelFormat.Format24bppRgb);
            using (var gm = Graphics.FromImage(output)) { gm.InterpolationMode = InterpolationMode.HighQualityBicubic; gm.DrawImage(sm, 0, 0, source.Width, source.Height); }
            return output;
        }

        private static int DetectHairTop(Bitmap mask, float fcx, float fcy, float chinY)
        {
            int scanL = Math.Clamp((int)(fcx - 50), 0, mask.Width - 1);
            int scanR = Math.Clamp((int)(fcx + 50), scanL, mask.Width - 1);
            int startY = (int)fcy;
            int yLim = Math.Max(0, (int)(fcy - (chinY - fcy)));
            
            var md = mask.LockBits(new System.Drawing.Rectangle(0,0,mask.Width,mask.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            int top = startY;
            unsafe {
                byte* ptr = (byte*)md.Scan0;
                for (int y = startY; y >= yLim; y--) {
                    bool hair = false;
                    for (int x = scanL; x <= scanR; x++) if (ptr[(y*md.Stride)+(x*3)] > 100) { hair = true; break; }
                    if (hair) top = y;
                }
            }
            mask.UnlockBits(md);
            return top;
        }

        private static Bitmap GenerateSheet(Bitmap base2x2, PrintLayout layout, int dpi, BackgroundColorOption bg)
        {
            int t = base2x2.Width;
            int gutter = 30; // 0.05 inch bleed safe gutter
            
            int w = 4*dpi, h = 6*dpi, cols = 2, rows = 3;
            int sX = (w - (cols*t + (cols-1)*gutter))/2;
            int sY = (h - (rows*t + (rows-1)*gutter))/2;

            if (layout == PrintLayout.Print4x8) { h = 8*dpi; rows = 4; sY = (h - (rows*t + (rows-1)*gutter))/2; }
            if (layout == PrintLayout.Print5x10) { w = 5*dpi; h = 10*dpi; rows = 4; sX = (w - (cols*t + (cols-1)*gutter))/2; sY = (h - (rows*t + (rows-1)*gutter))/2; }
            
            var sheet = new Bitmap(w, h, PixelFormat.Format24bppRgb);
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

            // Inject ICC Profile if exists
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
            lock(_modelLock) { _session?.Dispose(); _detector?.Dispose(); _predictor?.Dispose(); _isDisposed = true; } 
            _masterOriginal?.Dispose(); _masterAlpha?.Dispose();
        }
    }
}
