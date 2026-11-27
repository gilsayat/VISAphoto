using Microsoft.Win32;
using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using USVisaPhotoAutomator.Services;
using System.Linq;

namespace USVisaPhotoAutomator
{
    public partial class MainWindow : Window
    {
        private readonly VisaPhotoProcessor _processor;
        private string? _lastInputPath;
        private string? _lastOutputPath;
        
        private TaskCompletionSource<string>? _aiCompletionSource;
        private bool _isEngineReady = false;

        private bool _isDragging = false;
        private Point _dragStartPos;
        private double _panX = 0, _panY = 0;
        private double _zoomScale = 0; 
        private double _tiltAngle = 0;
        private bool _isShiftDown = false;

        public MainWindow()
        {
            InitializeComponent();
            _processor = new VisaPhotoProcessor();
            InitializeWebEngineAsync();
        }

        private async void InitializeWebEngineAsync()
        {
            // Initializes the invisible Edge browser
            await AiBrowser.EnsureCoreWebView2Async(null);
            
            // Map our local wwwroot folder to a virtual URL so it functions offline
            AiBrowser.CoreWebView2.SetVirtualHostNameToFolderMapping(
                "local.appassets", 
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "wwwroot"), 
                Microsoft.Web.WebView2.Core.CoreWebView2HostResourceAccessKind.Allow);
            
            AiBrowser.WebMessageReceived += AiBrowser_WebMessageReceived;
            AiBrowser.Source = new Uri("http://local.appassets/index.html");
        }

        private void AiBrowser_WebMessageReceived(object? sender, Microsoft.Web.WebView2.Core.CoreWebView2WebMessageReceivedEventArgs e)
        {
            string json = e.TryGetWebMessageAsString();
            if (json.Contains("\"READY\"")) {
                _isEngineReady = true;
                UpdateStatus("WebGPU Engine Active and Offline.", "#00FF7F");
            } else if (_aiCompletionSource != null) {
                _aiCompletionSource.SetResult(json);
            }
        }

        // The Bridge: Passes the photo to JS, waits for the JS WebGPU to return the transparent PNG
        private async Task<byte[]> ExtractBackgroundViaWebEngine(string imagePath)
        {
            _aiCompletionSource = new TaskCompletionSource<string>();
            
            byte[] imgBytes = await File.ReadAllBytesAsync(imagePath);
            string base64In = "data:image/jpeg;base64," + Convert.ToBase64String(imgBytes);
            
            // Fire the huge Base64 string directly into the JS runtime
            AiBrowser.CoreWebView2.PostWebMessageAsString(base64In);
            
            // Wait for JS to compute the ViTMatte model and respond
            string responseJson = await _aiCompletionSource.Task;
            _aiCompletionSource = null;
            
            if (responseJson.Contains("\"success\":true")) {
                // Parse out the returned Base64 PNG
                int dataIndex = responseJson.IndexOf("\"data\":\"") + 8;
                string base64Out = responseJson.Substring(dataIndex).Split('"')[0];
                string cleanBase64 = base64Out.Replace("data:image/png;base64,", "");
                return Convert.FromBase64String(cleanBase64);
            }
            throw new Exception("WebGPU Engine failed to extract the background.");
        }

        // -------------------------------------------------------------------------
        // STANDARD UI EVENT HANDLERS
        // -------------------------------------------------------------------------
        private void Window_PreviewKeyDown(object sender, KeyEventArgs e) { if (e.Key == Key.LeftShift || e.Key == Key.RightShift) _isShiftDown = true; }
        private void Window_PreviewKeyUp(object sender, KeyEventArgs e) { if (e.Key == Key.LeftShift || e.Key == Key.RightShift) _isShiftDown = false; }
        private void DropZone_DragEnter(object sender, DragEventArgs e) { if (e.Data.GetDataPresent(DataFormats.FileDrop)) { e.Effects = DragDropEffects.Copy; CanvasBorder.BorderBrush = new SolidColorBrush(Colors.DodgerBlue); } }
        private void DropZone_DragLeave(object sender, DragEventArgs e) { CanvasBorder.BorderBrush = new SolidColorBrush(Colors.Transparent); }
        
        private void DropZone_Drop(object sender, DragEventArgs e)
        {
            CanvasBorder.BorderBrush = new SolidColorBrush(Colors.Transparent);
            if (e.Data.GetDataPresent(DataFormats.FileDrop)) HandleImageSelection(((string[])e.Data.GetData(DataFormats.FileDrop)).FirstOrDefault());
        }

        private void BtnBrowse_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog { Filter = "Image Files|*.jpg;*.jpeg;*.png" };
            if (openFileDialog.ShowDialog() == true) HandleImageSelection(openFileDialog.FileName);
        }

        private async void HandleImageSelection(string? filePath)
        {
            if (string.IsNullOrEmpty(filePath) || !_isEngineReady) return;
            _lastInputPath = filePath; _lastOutputPath = null;
            ResetManualOffsets();

            BtnGenerate.IsEnabled = false;
            UpdateStatus("WebGPU Interop: Running Vision Transformer...", "#FFD700");
            ProcessingProgress.Visibility = Visibility.Visible;
            PlaceholderUI.Visibility = Visibility.Collapsed;
            PreviewImage.Visibility = Visibility.Collapsed;
            CompliancePanel.Visibility = Visibility.Collapsed;

            try 
            {
                // 1. Let the Invisible Browser rip the background out natively
                byte[] transparentPngBytes = await ExtractBackgroundViaWebEngine(filePath);
                
                // 2. Hand BOTH the original (for facial landmarks) and the transparent PNG to the C# Layout Engine
                await Task.Run(() => _processor.InitializeMasterFile(filePath, transparentPngBytes));
                
                CanvasBorder.Background = (CmbBgColor.SelectedIndex == 0) ? new SolidColorBrush(Colors.White) : new SolidColorBrush(Color.FromRgb(242, 242, 242));
                VisaBlueprint.Visibility = Visibility.Visible;
                CompliancePanel.Visibility = Visibility.Visible;
                BtnGenerate.IsEnabled = true;
                BtnOpenFolder.Visibility = Visibility.Collapsed;
                UpdateStatus("Ready. Drag to Pan, Scroll to Zoom.", "#00FF7F");
                
                TriggerLiveRender();
            } 
            catch (Exception ex) 
            {
                UpdateStatus($"Extraction Failed: {ex.Message}", "#FF5C5C");
                PlaceholderUI.Visibility = Visibility.Visible;
            }
            finally { ProcessingProgress.Visibility = Visibility.Collapsed; }
        }

        private void Canvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            _isDragging = true; _dragStartPos = e.GetPosition(CanvasBorder); CanvasBorder.CaptureMouse();
        }

        private void Canvas_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            _isDragging = false; CanvasBorder.ReleaseMouseCapture(); TriggerLiveRender(); 
        }

        private void Canvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (!_isDragging) return;
            Point currentPos = e.GetPosition(CanvasBorder);
            _panX += (currentPos.X - _dragStartPos.X) / 600.0 * 100.0;
            _panY += (currentPos.Y - _dragStartPos.Y) / 600.0 * 100.0;
            _dragStartPos = currentPos; TriggerLiveRender();
        }

        private void Canvas_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (_isShiftDown) {
                _tiltAngle = Math.Clamp(_tiltAngle + (e.Delta > 0 ? 0.5 : -0.5), -25, 25);
            } else {
                double zoomDelta = e.Delta > 0 ? 2.0 : -2.0; 
                _zoomScale = Math.Clamp(_zoomScale + zoomDelta, -20, 40);
                Point cursor = e.GetPosition(CanvasBorder);
                double cursorNormX = (cursor.X - 300.0) / 600.0; 
                double cursorNormY = (cursor.Y - 300.0) / 600.0;
                if (zoomDelta > 0) { _panX -= cursorNormX * 2.0; _panY -= cursorNormY * 2.0; }
                else { _panX += cursorNormX * 2.0; _panY += cursorNormY * 2.0; }
            }
            TriggerLiveRender();
        }

        private void Canvas_MouseLeave(object sender, MouseEventArgs e) { if (_isDragging) Canvas_MouseLeftButtonUp(sender, null!); }

        private void ResetManualOffsets() { _panX = 0; _panY = 0; _zoomScale = 0; _tiltAngle = 0; TriggerLiveRender(); }
        private void BtnSnapAuto_Click(object sender, RoutedEventArgs e) { ResetManualOffsets(); }
        private void Setting_Changed(object sender, RoutedEventArgs e) { 
            if (CanvasBorder != null && CmbBgColor != null) CanvasBorder.Background = (CmbBgColor.SelectedIndex == 0) ? new SolidColorBrush(Colors.White) : new SolidColorBrush(Color.FromRgb(242, 242, 242));
            TriggerLiveRender(); 
        }

        private async void TriggerLiveRender()
        {
            byte[] frame = await Task.Run(() => _processor.RenderFrame(GetCurrentRequest(), previewMode: true));
            ShowImageFromBytes(frame);
            UpdateComplianceUI(_processor.ComputeCompliance(GetCurrentRequest()));
        }

        private void UpdateComplianceUI(ComplianceReport rep)
        {
            Dispatcher.Invoke(() => {
                ValHeadHeight.Text = rep.HeadHeight.ValueText; StatHeadHeight.Text = rep.HeadHeight.Pass ? "PASS" : "ADJUST"; StatHeadHeight.Foreground = rep.HeadHeight.Pass ? new SolidColorBrush(Color.FromRgb(16, 185, 129)) : new SolidColorBrush(Color.FromRgb(245, 158, 11));
                ValEyeLevel.Text = rep.EyeLevel.ValueText; StatEyeLevel.Text = rep.EyeLevel.Pass ? "PASS" : "ADJUST"; StatEyeLevel.Foreground = rep.EyeLevel.Pass ? new SolidColorBrush(Color.FromRgb(16, 185, 129)) : new SolidColorBrush(Color.FromRgb(245, 158, 11));
                ValCentering.Text = rep.Centering.ValueText; StatCentering.Text = rep.Centering.Pass ? "PASS" : "ADJUST"; StatCentering.Foreground = rep.Centering.Pass ? new SolidColorBrush(Color.FromRgb(16, 185, 129)) : new SolidColorBrush(Color.FromRgb(245, 158, 11));
                ValHeadLevel.Text = rep.HeadTilt.ValueText; StatHeadLevel.Text = rep.HeadTilt.Pass ? "PASS" : "ADJUST"; StatHeadLevel.Foreground = rep.HeadTilt.Pass ? new SolidColorBrush(Color.FromRgb(16, 185, 129)) : new SolidColorBrush(Color.FromRgb(245, 158, 11));
                LblOverallPass.Text = rep.OverallPass ? "PASS" : "ADJUST"; LblOverallPass.Foreground = rep.OverallPass ? new SolidColorBrush(Color.FromRgb(16, 185, 129)) : new SolidColorBrush(Color.FromRgb(245, 158, 11));
            });
        }

        private ProcessingRequest GetCurrentRequest()
        {
            return new ProcessingRequest {
                Layout = CmbLayout.SelectedIndex switch { 0 => PrintLayout.Print4x6, 1 => PrintLayout.Print4x8, 2 => PrintLayout.Print5x10, _ => PrintLayout.Digital2x2 },
                BgColor = CmbBgColor.SelectedIndex == 0 ? BackgroundColorOption.White : BackgroundColorOption.OffWhite,
                Tilt = _tiltAngle, ZoomOffset = _zoomScale, PanX = _panX, PanY = _panY,
            };
        }

        private async void BtnGenerate_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_lastInputPath)) return;
            
            SaveFileDialog saveFileDialog = new SaveFileDialog { Filter = "JPEG Image|*.jpg|All Files|*.*", Title = "Save U.S. Visa Photo Print Sheet", FileName = $"{Path.GetFileNameWithoutExtension(_lastInputPath)}_VisaReady.jpg", InitialDirectory = Path.GetDirectoryName(_lastInputPath) ?? AppDomain.CurrentDomain.BaseDirectory };
            if (saveFileDialog.ShowDialog() != true) return;
            string outputPath = saveFileDialog.FileName;

            BtnGenerate.IsEnabled = false;
            try
            {
                UpdateStatus("Compiling 600 DPI Studio Output...", "#0066FF");
                ProcessingProgress.Visibility = Visibility.Visible;
                var req = GetCurrentRequest(); req.OutputPath = outputPath;

                await Task.Run(() => _processor.GenerateHighResPrint(req));
                _lastOutputPath = outputPath;
                
                UpdateStatus($"Generation Complete!", "#00AA00");
                BtnOpenFolder.Visibility = Visibility.Visible;
            }
            catch (Exception ex) { UpdateStatus($"Error: {ex.Message}", "#FF0000"); }
            finally { ProcessingProgress.Visibility = Visibility.Collapsed; BtnGenerate.IsEnabled = true; }
        }

        private void ShowImageFromBytes(byte[] bytes)
        {
            try {
                BitmapImage bitmap = new BitmapImage();
                using (var stream = new MemoryStream(bytes)) {
                    bitmap.BeginInit(); bitmap.CacheOption = BitmapCacheOption.OnLoad; bitmap.StreamSource = stream; bitmap.EndInit();
                }
                bitmap.Freeze();
                Dispatcher.Invoke(() => { PreviewImage.Source = bitmap; PreviewImage.Visibility = Visibility.Visible; });
            } catch {}
        }

        private void UpdateStatus(string message, string hexColor) { Dispatcher.Invoke(() => { LblStatus.Text = message; LblStatus.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString(hexColor)); }); }
        private void BtnOpenFolder_Click(object sender, RoutedEventArgs e) { if (!string.IsNullOrEmpty(_lastOutputPath) && File.Exists(_lastOutputPath)) Process.Start("explorer.exe", $"/select,\"{_lastOutputPath}\""); }
    }
}
