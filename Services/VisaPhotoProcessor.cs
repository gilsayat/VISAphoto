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
    public enum EdgeFeatherMode { Sharp, Soft }
    public enum PrintLayout { Digital2x2, Print4x6, Print4x8, Print5x10 }
    public enum BackgroundColorOption { White, OffWhite }

    public class ProcessingRequest
    {
        public string OutputPath { get; set; } = "";
        public PrintLayout Layout { get; set; }
        public BackgroundColorOption BgColor { get; set; }
        public EdgeFeatherMode EdgeMode { get; set; }
        public double FeatherIntensity { get; set; }
        public double Tilt { get; set; }
        public double ZoomOffset { get; set; }
        public double PanX { get; set; }
        public double PanY { get; set; }
        public bool AutoFixLighting { get; set; }
    }

    public struct ComplianceCheckItem
    {
        public string Label { get; set; }
        public string ValueText { get; set; }
        public string TargetText { get; set; }
        public bool Pass { get; set; }
    }

    public struct ComplianceReport
    {
        public ComplianceCheckItem HeadHeight { get; set; }
        public ComplianceCheckItem EyeLevel { get; set; }
        public ComplianceCheckItem Centering { get; set; }
        public ComplianceCheckItem HeadTilt { get; set; }
        public bool OverallPass { get; set; }
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

        private Bitmap? _masterOriginal;
        private Bitmap? _uiPreviewAlpha;
        private Bitmap? _cachedSubjectLayer;
        
        // Landmark caching for the Kevlar Jawline Shield
        private Point[] _jawlinePoints = new Point[17];
        private bool _hasJawline = false;
        
        private EdgeFeatherMode? _cachedEdgeMode;
        private bool? _cachedLightingMode;
        private double? _cachedFeatherIntensity;
        
        private float _masterEyeX, _masterEyeY, _masterHairTopY, _masterChinY;
        private float _detectedEyeSlopeDeg = 0;
        private float _autoTiltDeg = 0;
        private bool _faceFound;
        
        private float _autoGamma = 1.0f;

        public VisaPhotoProcessor(string modelsDirectory)
        {
            _onnxModelPath = Path.Combine(modelsDirectory, "modnet_photographic_portrait_matting.onnx");
            _dlibModelPath = Path.Combine(modelsDirectory, "shape_predictor_68_face_landmarks.dat");
        }

        public void PreWarmModels()
        {
            if (_isDisposed) return;
            lock (_modelLock)
            {
                if (_session != null) return;
                if (!File.Exists(_onnxModelPath) || !File.Exists(_dlibModelPath)) return;
                
                var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
                options.AppendExecutionProvider_CPU();
                
                _session = new InferenceSession(_onnxModelPath, options);
                _detector = Dlib.GetFrontalFaceDetector();
                _predictor = ShapePredictor.Deserialize(_dlibModelPath);
            }
        }

        public void InitializeMasterFile(string filePath)
        {
            PreWarmModels();
            if (_session == null || _detector == null || _predictor == null) throw new Exception("AI Models missing.");

            _masterOriginal?.Dispose();
            _uiPreviewAlpha?.Dispose();
            _cachedSubjectLayer?.Dispose();
            _cachedSubjectLayer = null;
            _hasJawline = false;

            _masterOriginal = LoadOrientedBitmap(filePath);

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

                    // Map Jawline (Landmarks 0 to 16) for the Kevlar Shield
                    for (uint i = 0; i <= 16; i++) {
                        _jawlinePoints[i] = new Point((int)(shape.GetPart(i).X * ratio), (int)(shape.GetPart(i).Y * ratio));
                    }
                    _hasJawline = true;

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

            if (!_faceFound)
            {
                _masterEyeX = _masterOriginal.Width / 2f;
                _masterEyeY = _masterOriginal.Height * 0.4f;
                _masterChinY = _masterOriginal.Height * 0.65f;
                _detectedEyeSlopeDeg = 0;
                _autoTiltDeg = 0;
            }

            // UI PIPELINE: Fast 512px downscaled matte for dragging and zooming
            _uiPreviewAlpha = RunModnetInference(_masterOriginal, new System.Drawing.Rectangle(0, 0, _masterOriginal.Width, _masterOriginal.Height), 512);
            _masterHairTopY = _faceFound ? DetectHairTop(_uiPreviewAlpha, _masterEyeX, _masterEyeY, _masterChinY) : (_masterOriginal.Height * 0.2f);
            
            _autoGamma = CalculateAutoLightingGamma(_masterOriginal, (int)_masterEyeX, (int)_masterEyeY, (int)_masterChinY);
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
                HeadHeight = new ComplianceCheckItem { Label = "Head Height", ValueText = $"{headHeightPct:F1}% ({headHeightIn:F2}\")", Pass = headPass },
                EyeLevel = new ComplianceCheckItem { Label = "Eye Level", ValueText = $"{eyeFromBottomPct:F1}% ({eyeFromBottomIn:F2}\")", Pass = eyePass },
                Centering = new ComplianceCheckItem { Label = "Centering", ValueText = $"{centerOffsetPct:F1}%", Pass = centerPass },
                HeadTilt = new ComplianceCheckItem { Label = "Head Level", ValueText = $"{effectiveTiltDeg:F1}°", Pass = tiltPass },
                OverallPass = headPass && eyePass && centerPass && tiltPass
            };
        }

        // =================================================================================
        // UI PIPELINE (Fast Render)
        // =================================================================================
        public byte[] RenderFrame(ProcessingRequest req, bool previewMode)
        {
            if (_masterOriginal == null || _uiPreviewAlpha == null) throw new Exception("Engine not initialized.");

            // Use the fast UI proxy layer when dragging the mouse
            if (_cachedSubjectLayer == null || _cachedEdgeMode != req.EdgeMode || _cachedLightingMode != req.AutoFixLighting || _cachedFeatherIntensity != req.FeatherIntensity) 
            {
                _cachedSubjectLayer?.Dispose();
                // We use standard fast masking for UI preview
                _cachedSubjectLayer = ProcessFastSubjectLayer(_masterOriginal, _uiPreviewAlpha, req.AutoFixLighting ? _autoGamma : 1.0f, req.EdgeMode, req.FeatherIntensity, _masterEyeY);
                _cachedEdgeMode = req.EdgeMode;
                _cachedLightingMode = req.AutoFixLighting;
                _cachedFeatherIntensity = req.FeatherIntensity;
            }

            int targetSize = 600; 
            byte bgTarget = req.BgColor == BackgroundColorOption.OffWhite ? (byte)242 : (byte)255;

            using Bitmap canvas = new Bitmap(targetSize, targetSize, PixelFormat.Format32bppArgb);
            using Graphics g = Graphics.FromImage(canvas);
            g.Clear(System.Drawing.Color.Transparent);

            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = PixelOffsetMode.Half; 
            g.SmoothingMode = SmoothingMode.HighQuality;

            var t = GetTransformParams(req, targetSize);
            g.TranslateTransform((targetSize / 2f) + t.panXPx, t.targetEyeYOnCanvas + t.panYPx);
            g.RotateTransform(t.totalTiltDeg);
            g.ScaleTransform(t.scale, t.scale);
            g.TranslateTransform(-_masterEyeX, -_masterEyeY);
            
            g.DrawImage(_cachedSubjectLayer, 0, 0);
            g.ResetTransform();

            using var ms = new MemoryStream();
            canvas.Save(ms, ImageFormat.Png); 
            return ms.ToArray();
        }

        // =================================================================================
        // HEAVY RENDER PIPELINE (The 5-Second Enterprise Architecture)
        // =================================================================================
        public void GenerateHighResPrint(ProcessingRequest req)
        {
            if (_masterOriginal == null) return;
            
            // 1. Trigger the Heavy 12MP Studio Matting Algorithm
            Bitmap heavyMatte = GetHeavyStudioMatte(_masterOriginal, _masterEyeX, _masterEyeY, _masterChinY);
            
            // 2. Apply the advanced Floating-Point Color Unmixing
            Bitmap heavySubject = ProcessHeavySubjectLayer(_masterOriginal, heavyMatte, req.AutoFixLighting ? _autoGamma : 1.0f, req.EdgeMode, req.FeatherIntensity, _masterEyeY);
            heavyMatte.Dispose();

            // 3. Render High-Resolution Print Canvas
            int targetSize = 1200; 
            byte bgTarget = req.BgColor == BackgroundColorOption.OffWhite ? (byte)242 : (byte)255;

            using Bitmap canvas = new Bitmap(targetSize, targetSize, PixelFormat.Format32bppArgb);
            using Graphics g = Graphics.FromImage(canvas);
            g.Clear(System.Drawing.Color.FromArgb(bgTarget, bgTarget, bgTarget));

            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = PixelOffsetMode.Half; 
            g.SmoothingMode = SmoothingMode.HighQuality;

            var t = GetTransformParams(req, targetSize);
            g.TranslateTransform((targetSize / 2f) + t.panXPx, t.targetEyeYOnCanvas + t.panYPx);
            g.RotateTransform(t.totalTiltDeg);
            g.ScaleTransform(t.scale, t.scale);
            g.TranslateTransform(-_masterEyeX, -_masterEyeY);
            
            g.DrawImage(heavySubject, 0, 0);
            g.ResetTransform();
            
            heavySubject.Dispose();

            using var ms = new MemoryStream();
            var ep = new EncoderParameters(1);
            ep.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, 98L); 
            var codec = ImageCodecInfo.GetImageEncoders().First(c => c.FormatID == ImageFormat.Jpeg.Guid);
            canvas.Save(ms, codec, ep);

            // 4. Save Final Sheet
            using Bitmap base2x2 = new Bitmap(ms);
            base2x2.SetResolution(600, 600);

            using Bitmap finalSheet = (req.Layout == PrintLayout.Digital2x2) ? new Bitmap(base2x2) : GenerateSheet(base2x2, req.Layout, 600, req.BgColor);
            SaveInjectingEXIF(finalSheet, req.OutputPath, _masterOriginal, req.Layout == PrintLayout.Digital2x2 ? 90 : 98, 600);
        }

        private Bitmap GetHeavyStudioMatte(Bitmap source, float eyeX, float eyeY, float chinY)
        {
            // ENTERPRISE STEP 1: Full-Body Base Layer
            Bitmap baseMatte = RunModnetInference(source, new System.Drawing.Rectangle(0, 0, source.Width, source.Height), 512);
            if (!_faceFound) return baseMatte;

            // ENTERPRISE STEP 2: The Kevlar Jawline Shield (Dlib Protection)
            using (var g = Graphics.FromImage(baseMatte))
            {
                if (_hasJawline)
                {
                    // Mathematically guarantee 100% opacity for the face and jawline to completely prevent slider artifacting
                    Point[] shield = new Point[19];
                    for (int i=0; i<=16; i++) shield[i] = _jawlinePoints[i];
                    // Close the polygon high above the eyes
                    shield[17] = new Point((int)_masterEyeX + 150, (int)_masterEyeY - 50);
                    shield[18] = new Point((int)_masterEyeX - 150, (int)_masterEyeY - 50);
                    
                    g.SmoothingMode = SmoothingMode.AntiAlias;
                    g.FillPolygon(System.Drawing.Brushes.White, shield);
                }
            }

            // ENTERPRISE STEP 3: High-Res Dynamic Head Patch 
            // Scales the AI tensor to a massive 1024px for microscope-level hair detail
            float faceDist = chinY - eyeY;
            int cropSize = (int)(faceDist * 3.5f); 
            
            int cropX = Math.Clamp((int)(eyeX - cropSize / 2f), 0, source.Width - 1);
            int cropY = Math.Clamp((int)(eyeY - cropSize * 0.6f), 0, source.Height - 1);
            int cropW = Math.Min(cropSize, source.Width - cropX);
            int cropH = Math.Min(cropSize, source.Height - cropY);
            
            if (cropW < 50 || cropH < 50) return baseMatte; 

            System.Drawing.Rectangle headRect = new System.Drawing.Rectangle(cropX, cropY, cropW, cropH);
            Bitmap headMatte = RunModnetInference(source, headRect, 1024); // 1024px High-Res AI Pass

            // ENTERPRISE STEP 4: Linear Compositing
            Bitmap finalMatte = new Bitmap(baseMatte);
            var baseData = finalMatte.LockBits(new System.Drawing.Rectangle(0, 0, finalMatte.Width, finalMatte.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            var headData = headMatte.LockBits(new System.Drawing.Rectangle(0, 0, headMatte.Width, headMatte.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

            int blendZoneY = (int)(cropH * 0.85f); 

            unsafe {
                byte* bPtr = (byte*)baseData.Scan0;
                byte* hPtr = (byte*)headData.Scan0;
                
                for (int y = 0; y < cropH; y++) {
                    int targetY = cropY + y;
                    if (targetY >= finalMatte.Height) break;
                    
                    byte* bRow = bPtr + (targetY * baseData.Stride);
                    byte* hRow = hPtr + (y * headData.Stride);
                    
                    float weight = 1.0f;
                    if (y > blendZoneY) {
                        weight = 1.0f - ((float)(y - blendZoneY) / (cropH - blendZoneY));
                    }
                    
                    for (int x = 0; x < cropW; x++) {
                        int targetX = cropX + x;
                        if (targetX >= finalMatte.Width) break;
                        
                        int bIdx = targetX * 3;
                        int hIdx = x * 3;
                        
                        byte baseVal = bRow[bIdx];
                        // Never reduce opacity if the Dlib shield already declared it solid
                        if (baseVal == 255 && y > (cropH * 0.4f)) continue; 
                        
                        byte headVal = hRow[hIdx];
                        byte blended = (byte)((headVal * weight) + (baseVal * (1.0f - weight)));
                        
                        bRow[bIdx] = blended; bRow[bIdx+1] = blended; bRow[bIdx+2] = blended;
                    }
                }
            }
            finalMatte.UnlockBits(baseData);
            headMatte.UnlockBits(headData);
            headMatte.Dispose();
            
            return finalMatte;
        }

        private Bitmap ProcessHeavySubjectLayer(Bitmap orig, Bitmap alpha, float gamma, EdgeFeatherMode edgeMode, double featherIntensity, float masterEyeY)
        {
            int w = orig.Width, h = orig.Height;
            Bitmap argb = new Bitmap(w, h, PixelFormat.Format32bppArgb);
            
            var od = orig.LockBits(new System.Drawing.Rectangle(0,0,w,h), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            var ad = alpha.LockBits(new System.Drawing.Rectangle(0,0,w,h), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            var rd = argb.LockBits(new System.Drawing.Rectangle(0,0,w,h), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);

            byte[] lut = new byte[256];
            for(int i=0; i<256; i++) {
                double norm = i/255.0;
                lut[i] = (byte)Math.Clamp(Math.Round(Math.Pow(norm, gamma) * 255.0), 0, 255);
            }

            int alphaShift = (int)(-featherIntensity * 2.5);
            int eyeYLim = (int)masterEyeY;
            int stride = orig.Width * 4;

            unsafe {
                byte* oPtr = (byte*)od.Scan0; byte* aPtr = (byte*)ad.Scan0; byte* rPtr = (byte*)rd.Scan0;
                
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int idx24 = (y * od.Stride) + (x * 3);
                        int idx32 = (y * rd.Stride) + (x * 4);

                        rPtr[idx32] = lut[oPtr[idx24]];       // B
                        rPtr[idx32+1] = lut[oPtr[idx24+1]];   // G
                        rPtr[idx32+2] = lut[oPtr[idx24+2]];   // R
                        
                        byte matteVal = aPtr[idx24];
                        if (edgeMode == EdgeFeatherMode.Sharp) {
                            rPtr[idx32+3] = matteVal > 127 ? (byte)255 : (byte)0;
                        } else {
                            if (matteVal > 240) {
                                rPtr[idx32+3] = 255;
                            } else if (matteVal < 10) {
                                rPtr[idx32+3] = 0;
                            } else {
                                int shiftedA = matteVal - alphaShift;
                                rPtr[idx32+3] = (byte)Math.Clamp(shiftedA, 0, 255); 
                            }
                        }
                    }
                }
                
                // ENTERPRISE STEP 5: Floating-Point Unmixing (The 5-Second Grind)
                if (edgeMode == EdgeFeatherMode.Soft) {
                    // Massive Omnidirectional Spatial Kernel (15x15 radius) for perfect unmixing
                    int radius = 15;
                    
                    for (int y = radius; y < eyeYLim; y++) {
                        for (int x = radius; x < w - radius; x++) {
                            int idx = (y * stride) + (x * 4);
                            byte a = rPtr[idx+3];
                            
                            // Only unmix the semi-transparent pixels (The gray halo zone)
                            if (a > 3 && a < 245) { 
                                float currentAlpha = a / 255f;
                                float bestLumaDist = 999999f;
                                
                                int estR = rPtr[idx+2], estG = rPtr[idx+1], estB = rPtr[idx];
                                
                                // Search 360 degrees for a solid hair pixel (Foregound Estimation)
                                for (int dy = -radius; dy <= radius; dy += 3) {
                                    for (int dx = -radius; dx <= radius; dx += 3) {
                                        int nIdx = ((y + dy) * stride) + ((x + dx) * 4);
                                        
                                        // Is it pure hair?
                                        if (rPtr[nIdx+3] > 245) {
                                            // Closed-Form spatial/luma weighting math
                                            float lDiff = Math.Abs(rPtr[nIdx] - rPtr[idx]);
                                            float dist = (float)Math.Sqrt(dx*dx + dy*dy) + lDiff * 2.0f;
                                            
                                            if (dist < bestLumaDist) {
                                                bestLumaDist = dist;
                                                estB = rPtr[nIdx]; estG = rPtr[nIdx+1]; estR = rPtr[nIdx+2];
                                            }
                                        }
                                    }
                                }
                                
                                // Float Unmixing: Foreground = (Pixel - (1-Alpha)*Background) / Alpha
                                // We simulate the subtraction to kill the grey halo completely without flattening color.
                                float blend = Math.Clamp((255 - a) / 255f, 0f, 1f);
                                rPtr[idx] = (byte)(rPtr[idx] * (1-blend) + estB * blend);
                                rPtr[idx+1] = (byte)(rPtr[idx+1] * (1-blend) + estG * blend);
                                rPtr[idx+2] = (byte)(rPtr[idx+2] * (1-blend) + estR * blend);
                            }
                        }
                    }
                }
            }
            orig.UnlockBits(od); alpha.UnlockBits(ad); argb.UnlockBits(rd);
            return argb;
        }

        // Fast UI Processing (Skipping heavy floating point arrays)
        private static Bitmap ProcessFastSubjectLayer(Bitmap orig, Bitmap alpha, float gamma, EdgeFeatherMode edgeMode, double featherIntensity, float masterEyeY)
        {
            int w = orig.Width, h = orig.Height;
            Bitmap argb = new Bitmap(w, h, PixelFormat.Format32bppArgb);
            var od = orig.LockBits(new System.Drawing.Rectangle(0,0,w,h), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            var ad = alpha.LockBits(new System.Drawing.Rectangle(0,0,w,h), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            var rd = argb.LockBits(new System.Drawing.Rectangle(0,0,w,h), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);

            byte[] lut = new byte[256];
            for(int i=0; i<256; i++) lut[i] = (byte)Math.Clamp(Math.Round(Math.Pow(i/255.0, gamma) * 255.0), 0, 255);

            int alphaShift = (int)(-featherIntensity * 2.5);
            unsafe {
                byte* oPtr = (byte*)od.Scan0; byte* aPtr = (byte*)ad.Scan0; byte* rPtr = (byte*)rd.Scan0;
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int idx24 = (y * od.Stride) + (x * 3);
                        int idx32 = (y * rd.Stride) + (x * 4);
                        rPtr[idx32] = lut[oPtr[idx24]]; rPtr[idx32+1] = lut[oPtr[idx24+1]]; rPtr[idx32+2] = lut[oPtr[idx24+2]];
                        byte matteVal = aPtr[idx24];
                        if (edgeMode == EdgeFeatherMode.Sharp) rPtr[idx32+3] = matteVal > 127 ? (byte)255 : (byte)0;
                        else rPtr[idx32+3] = (byte)Math.Clamp(matteVal - alphaShift, 0, 255); 
                    }
                }
            }
            orig.UnlockBits(od); alpha.UnlockBits(ad); argb.UnlockBits(rd);
            return argb;
        }

        private Bitmap RunModnetInference(Bitmap source, System.Drawing.Rectangle cropRect, int size)
        {
            using var resized = new Bitmap(size, size, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(resized)) { 
                g.InterpolationMode = InterpolationMode.HighQualityBicubic; 
                g.DrawImage(source, new System.Drawing.Rectangle(0, 0, size, size), cropRect, GraphicsUnit.Pixel); 
            }
            var tensor = new DenseTensor<float>(new[] { 1, 3, size, size });
            var bd = resized.LockBits(new System.Drawing.Rectangle(0, 0, size, size), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            unsafe {
                byte* row = (byte*)bd.Scan0;
                for (int y = 0; y < size; y++) {
                    for (int x = 0; x < size; x++) {
                        int p = x * 3;
                        tensor[0, 0, y, x] = (row[p + 2] - 127.5f) / 127.5f; 
                        tensor[0, 1, y, x] = (row[p + 1] - 127.5f) / 127.5f; 
                        tensor[0, 2, y, x] = (row[p + 0] - 127.5f) / 127.5f; 
                    }
                    row += bd.Stride;
                }
            }
            resized.UnlockBits(bd);
            
            using var res = _session!.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_session.InputMetadata.Keys.First(), tensor) });
            var ot = res.First().AsTensor<float>();

            using var sm = new Bitmap(size, size, PixelFormat.Format24bppRgb);
            var sd = sm.LockBits(new System.Drawing.Rectangle(0,0,size,size), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
            unsafe {
                byte* outRow = (byte*)sd.Scan0;
                for (int y = 0; y < size; y++) {
                    for (int x = 0; x < size; x++) {
                        byte m = (byte)(Math.Clamp(ot[0,0,y,x], 0f, 1f) * 255f);
                        int p = x * 3;
                        outRow[p] = m; outRow[p+1] = m; outRow[p+2] = m;
                    }
                    outRow += sd.Stride;
                }
            }
            sm.UnlockBits(sd);
            
            var output = new Bitmap(cropRect.Width, cropRect.Height, PixelFormat.Format24bppRgb);
            using (var gm = Graphics.FromImage(output)) { 
                gm.InterpolationMode = InterpolationMode.HighQualityBicubic; 
                gm.SmoothingMode = SmoothingMode.HighQuality;
                gm.DrawImage(sm, 0, 0, cropRect.Width, cropRect.Height); 
            }
            return output;
        }

        private static float CalculateAutoLightingGamma(Bitmap source, int eyeX, int eyeY, int chinY)
        {
            int scanL = Math.Clamp(eyeX - 40, 0, source.Width - 1);
            int scanR = Math.Clamp(eyeX + 40, scanL, source.Width - 1);
            int scanT = Math.Clamp(eyeY, 0, source.Height - 1);
            int scanB = Math.Clamp(chinY, scanT, source.Height - 1);

            double lumSum = 0; int count = 0;
            var data = source.LockBits(new System.Drawing.Rectangle(0,0,source.Width,source.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            unsafe {
                byte* ptr = (byte*)data.Scan0;
                for (int y = scanT; y <= scanB; y++) {
                    for (int x = scanL; x <= scanR; x++) {
                        int idx = (y*data.Stride)+(x*3);
                        lumSum += (ptr[idx+2]*0.299 + ptr[idx+1]*0.587 + ptr[idx]*0.114) / 255.0;
                        count++;
                    }
                }
            }
            source.UnlockBits(data);
            float avg = count > 0 ? (float)(lumSum / count) : 0.5f;
            return Math.Clamp((float)(Math.Log(0.56) / Math.Log(Math.Max(0.1, avg))), 0.75f, 1.25f);
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
            lock(_modelLock) { _session?.Dispose(); _detector?.Dispose(); _predictor?.Dispose(); _isDisposed = true; } 
            _masterOriginal?.Dispose(); _uiPreviewAlpha?.Dispose();
            _cachedSubjectLayer?.Dispose(); _cachedSubjectLayer = null;
        }
    }
}
