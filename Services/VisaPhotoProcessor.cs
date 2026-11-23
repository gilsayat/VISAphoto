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
        public string InputPath { get; set; } = "";
        public string OutputPath { get; set; } = "";
        public PrintLayout Layout { get; set; }
        public BackgroundColorOption BgColor { get; set; }
        public EdgeFeatherMode EdgeMode { get; set; }
        public double Tilt { get; set; }
        public double ZoomOffset { get; set; }
        public double PanX { get; set; }
        public double PanY { get; set; }
        public double Brightness { get; set; }
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

        public void ProcessVisaPhoto(ProcessingRequest req)
        {
            PreWarmModels();
            if (_session == null || _detector == null || _predictor == null) throw new Exception("Models missing.");

            int dpi = 600; // Always master render at 600 DPI for flawless edges
            int targetSize = 2 * dpi; // 1200x1200px
            byte bgTarget = req.BgColor == BackgroundColorOption.OffWhite ? (byte)242 : (byte)255;

            using Bitmap originalImage = LoadOrientedBitmap(req.InputPath);
            
            // 1. Transform Matrix (Tilt, Color Temp, Brightness)
            using Bitmap preppedImage = ApplyColorAndTransform(originalImage, req);

            // 2. Dlib Landmark Anchor
            int faceLeft, faceRight, browLineY, chinBottomY;
            float eyeLineY, faceCenterX;
            
            string tempFile = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".bmp");
            preppedImage.Save(tempFile, ImageFormat.Bmp);
            try
            {
                using var dlibImg = Dlib.LoadImage<RgbPixel>(tempFile);
                var faces = _detector.Operator(dlibImg, 0);
                if (faces.Length == 0) faces = _detector.Operator(dlibImg, 1);
                if (faces.Length == 0) throw new Exception("No face detected after manual adjustments.");
                
                var face = faces.OrderByDescending(r => r.Width * r.Height).First();
                using var shape = _predictor.Detect(dlibImg, face);

                faceLeft = (int)face.Left; faceRight = (int)face.Right; browLineY = (int)face.Top;
                chinBottomY = Math.Max((int)shape.GetPart(8).Y, (int)shape.GetPart(9).Y);
                
                float lY = 0, rY = 0, lX = 0, rX = 0;
                for (uint i=36; i<=41; i++) { lX += shape.GetPart(i).X; lY += shape.GetPart(i).Y; }
                for (uint i=42; i<=47; i++) { rX += shape.GetPart(i).X; rY += shape.GetPart(i).Y; }
                eyeLineY = (lY/6f + rY/6f)/2f;
                faceCenterX = (((lX/6f + rX/6f)/2f) + shape.GetPart(27).X)/2f;
            }
            finally { if (File.Exists(tempFile)) File.Delete(tempFile); }

            // 3. ROI Matting & Decontamination
            using Bitmap coarseMask = GetCoarseMask(preppedImage);
            int hairTopY = DetectHairTop(coarseMask, faceLeft, faceRight, browLineY, chinBottomY, faceCenterX);
            
            // Apply Edge Choke via Sub-Pixel Morphology before final roll-off
            using Bitmap refinedAlpha = RefineAlphaWithChoke(preppedImage, coarseMask, browLineY, req.EdgeMode, req.EdgeChoke);
            using Bitmap decontaminatedBg = Composite(preppedImage, refinedAlpha, bgTarget);

            // 4. Strict Framing with UI Manual Override Offsets
            int rawHeadHeight = chinBottomY - hairTopY;
            if (rawHeadHeight <= 20) throw new Exception("Head height unmeasurable.");

            // Base math is 56% head size, modified by the user's Zoom UI slider
            float baseScale = (targetSize * 0.56f) / rawHeadHeight;
            float scale = baseScale * (1.0f + (float)req.ZoomOffset / 100f);

            float scaledHairTopY = hairTopY * scale;
            float scaledEyeLineY = eyeLineY * scale;
            float eyeOffsetFromHairTop = scaledEyeLineY - scaledHairTopY;

            // Center subject + apply user's UI Pan offsets
            float desiredEyeFromTop = targetSize * (1.0f - 0.60f) + (float)req.PanY * 4f; 
            float desiredHairTopFromTop = desiredEyeFromTop - eyeOffsetFromHairTop;
            
            // Boundary hard-clamps to prevent illegal Visa floating head / low eyes
            if (desiredHairTopFromTop < targetSize * 0.05f) {
                desiredHairTopFromTop = targetSize * 0.05f;
                desiredEyeFromTop = desiredHairTopFromTop + eyeOffsetFromHairTop;
            }
            if ((targetSize - desiredEyeFromTop)/targetSize < 0.56f) {
                desiredEyeFromTop = targetSize * (1.0f - 0.56f);
            }

            int offsetX = (int)Math.Round((targetSize / 2.0f) - (faceCenterX * scale) + (req.PanX * 4f));
            int offsetY = (int)Math.Round(desiredEyeFromTop - scaledEyeLineY);

            int scaledW = Math.Max(1, (int)(decontaminatedBg.Width * scale));
            int scaledH = Math.Max(1, (int)(decontaminatedBg.Height * scale));

            using Bitmap base2x2 = new Bitmap(targetSize, targetSize, PixelFormat.Format24bppRgb);
            base2x2.SetResolution(dpi, dpi);
            using (Graphics g = Graphics.FromImage(base2x2))
            {
                g.Clear(System.Drawing.Color.FromArgb(bgTarget, bgTarget, bgTarget));
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.DrawImage(decontaminatedBg, offsetX, offsetY, scaledW, scaledH);
                
                if (req.Layout != PrintLayout.Digital2x2)
                    g.DrawRectangle(new Pen(System.Drawing.Color.Black, 2f), 4, 4, targetSize - 8, targetSize - 8);
            }

            // 5. Layout Engine
            using Bitmap finalSheet = (req.Layout == PrintLayout.Digital2x2) ? new Bitmap(base2x2) : GenerateSheet(base2x2, req.Layout, dpi, bgTarget);

            // 6. Output
            var codec = ImageCodecInfo.GetImageEncoders().FirstOrDefault(c => c.FormatID == ImageFormat.Jpeg.Guid);
            if (codec == null) { finalSheet.Save(req.OutputPath, ImageFormat.Jpeg); return; }
            using var ep = new EncoderParameters(1);
            ep.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, req.Layout == PrintLayout.Digital2x2 ? 90L : 98L);
            finalSheet.Save(req.OutputPath, codec, ep);
        }

        private static Bitmap ApplyColorAndTransform(Bitmap source, ProcessingRequest req)
        {
            var res = new Bitmap(source.Width, source.Height, PixelFormat.Format24bppRgb);
            using (Graphics g = Graphics.FromImage(res))
            {
                g.Clear(System.Drawing.Color.White);
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                if (Math.Abs(req.Tilt) > 0.1)
                {
                    g.TranslateTransform(source.Width / 2f, source.Height / 2f);
                    g.RotateTransform((float)req.Tilt);
                    g.TranslateTransform(-source.Width / 2f, -source.Height / 2f);
                }
                g.DrawImage(source, 0, 0);
            }
            
            if (Math.Abs(req.Brightness) < 1 && Math.Abs(req.ColorTemp) < 1) return res;

            var data = res.LockBits(new System.Drawing.Rectangle(0,0,res.Width,res.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            float brightMod = 1.0f + ((float)req.Brightness / 100f);
            float tempModR = 1.0f + ((float)req.ColorTemp / 150f);
            float tempModB = 1.0f - ((float)req.ColorTemp / 150f);

            unsafe
            {
                byte* ptr = (byte*)data.Scan0;
                for (int i = 0; i < res.Width * res.Height; i++)
                {
                    int idx = i * 3;
                    ptr[idx] = (byte)Math.Clamp(ptr[idx] * brightMod * tempModB, 0, 255);
                    ptr[idx+1] = (byte)Math.Clamp(ptr[idx+1] * brightMod, 0, 255);
                    ptr[idx+2] = (byte)Math.Clamp(ptr[idx+2] * brightMod * tempModR, 0, 255);
                }
            }
            res.UnlockBits(data);
            return res;
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

        private static int DetectHairTop(Bitmap mask, int fl, int fr, int ft, int cb, float fcx)
        {
            int scanL = Math.Clamp((int)(fcx - 30), 0, mask.Width - 1);
            int scanR = Math.Clamp((int)(fcx + 30), scanL, mask.Width - 1);
            int yLim = Math.Max(0, ft - (cb - ft));
            var md = mask.LockBits(new System.Drawing.Rectangle(0,0,mask.Width,mask.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            int top = ft;
            unsafe {
                byte* ptr = (byte*)md.Scan0;
                for (int y = ft; y >= yLim; y--) {
                    bool hair = false;
                    for (int x = scanL; x <= scanR; x++) if (ptr[(y*md.Stride)+(x*3)] > 100) { hair = true; break; }
                    if (hair) top = y;
                }
            }
            mask.UnlockBits(md);
            return top;
        }

        private static Bitmap RefineAlphaWithChoke(Bitmap guide, Bitmap mask, int browY, EdgeFeatherMode mode, int choke)
        {
            // Mathematical erosion/dilation proxy via thresholding before blur
            int w = guide.Width, h = guide.Height;
            var res = new Bitmap(w, h, PixelFormat.Format24bppRgb);
            var md = mask.LockBits(new System.Drawing.Rectangle(0,0,w,h), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            var rd = res.LockBits(new System.Drawing.Rectangle(0,0,w,h), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
            
            float chokeOffset = (float)choke * 0.05f; // -0.2 to +0.2 threshold shift
            
            unsafe {
                byte* mPtr = (byte*)md.Scan0; byte* rPtr = (byte*)rd.Scan0;
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int idx = (y*md.Stride)+(x*3);
                        float val = mPtr[idx]/255f;
                        
                        // Apply Choke
                        val = Math.Clamp(val + chokeOffset, 0f, 1f);
                        
                        // Harder threshold on bottom, softer on hair
                        float blend = Math.Clamp((y - browY) / 100f, 0f, 1f);
                        float low = 0.2f + (0.1f * blend);
                        float high = 0.8f - (0.1f * blend);
                        
                        float a = 0f;
                        if (val > high) a = 1f;
                        else if (val > low) { float t = (val - low)/(high - low); a = t*t*(3-2*t); }
                        
                        byte b = (byte)(a * 255f);
                        rPtr[idx] = b; rPtr[idx+1] = b; rPtr[idx+2] = b;
                    }
                }
            }
            mask.UnlockBits(md); res.UnlockBits(rd);
            return res;
        }

        private static Bitmap Composite(Bitmap src, Bitmap alpha, byte bgTarget)
        {
            var res = new Bitmap(src.Width, src.Height, PixelFormat.Format24bppRgb);
            var sd = src.LockBits(new System.Drawing.Rectangle(0,0,src.Width,src.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            var ad = alpha.LockBits(new System.Drawing.Rectangle(0,0,src.Width,src.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            var rd = res.LockBits(new System.Drawing.Rectangle(0,0,src.Width,src.Height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
            unsafe {
                byte* s = (byte*)sd.Scan0, a = (byte*)ad.Scan0, r = (byte*)rd.Scan0;
                for (int y = 0; y < src.Height; y++) {
                    for (int x = 0; x < src.Width; x++) {
                        int idx = (y*sd.Stride)+(x*3);
                        float aV = a[idx]/255f;
                        r[idx] = (byte)Math.Clamp(s[idx]*aV + bgTarget*(1-aV), 0, 255);
                        r[idx+1] = (byte)Math.Clamp(s[idx+1]*aV + bgTarget*(1-aV), 0, 255);
                        r[idx+2] = (byte)Math.Clamp(s[idx+2]*aV + bgTarget*(1-aV), 0, 255);
                    }
                }
            }
            src.UnlockBits(sd); alpha.UnlockBits(ad); res.UnlockBits(rd);
            return res;
        }

        private static Bitmap GenerateSheet(Bitmap base2x2, PrintLayout layout, int dpi, byte bgTarget)
        {
            int t = base2x2.Width;
            int w = 4*dpi, h = 6*dpi, rows = 3, sX = 0, sY = 0;
            if (layout == PrintLayout.Print4x8) { h = 8*dpi; rows = 4; }
            if (layout == PrintLayout.Print5x10) { w = 5*dpi; h = 10*dpi; rows = 4; sX = (int)(0.5f*dpi); sY = (int)(1.0f*dpi); }
            
            var sheet = new Bitmap(w, h, PixelFormat.Format24bppRgb);
            sheet.SetResolution(dpi, dpi);
            using Graphics gs = Graphics.FromImage(sheet);
            gs.Clear(System.Drawing.Color.White);
            for (int y = 0; y < rows; y++) for (int x = 0; x < 2; x++) gs.DrawImageUnscaled(base2x2, sX + x * t, sY + y * t);
            return sheet;
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
        
        public void Dispose() { lock(_modelLock) { _session?.Dispose(); _detector?.Dispose(); _predictor?.Dispose(); _isDisposed = true; } }
    }
}
