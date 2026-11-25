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
        
        private bool _isRendering = false;
        private bool _renderPending = false;
        private bool _isEngineReady = false;

        // Smart Canvas Variables
        private bool _isDragging = false;
        private Point _dragStartPos;
        private double _panX = 0, _panY = 0;
        private double _zoomScale = 0; 
        private double _tiltAngle = 0;
        private bool _isShiftDown = false;

        public MainWindow()
        {
            InitializeComponent();
            string modelsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models");
            _processor = new VisaPhotoProcessor(modelsDir);
            Task.Run(() => { try { _processor.PreWarmModels(); UpdateStatus("Engine loaded and ready.", "#00FF7F"); } catch {} });
        }

        private void Window_PreviewKeyDown(object sender, KeyEventArgs e) { if (e.Key == Key.LeftShift || e.Key == Key.RightShift) _isShiftDown = true; }
        private void Window_PreviewKeyUp(object sender, KeyEventArgs e) { if (e.Key == Key.LeftShift || e.Key == Key.RightShift) _isShiftDown = false; }

        private void DropZone_DragEnter(object sender, DragEventArgs e) { if (e.Data.GetDataPresent(DataFormats.FileDrop)) { e.Effects = DragDropEffects.Copy; CanvasBorder.BorderBrush = new SolidColorBrush(Colors.DodgerBlue); } }
        private void DropZone_DragLeave(object sender, DragEventArgs e) { CanvasBorder.BorderBrush = new SolidColorBrush(Colors.Transparent); }
        
        private void DropZone_Drop(object sender, DragEventArgs e)
        {
            CanvasBorder.BorderBrush = new SolidColorBrush(Colors.Transparent);
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                var file = ((string[])e.Data.GetData(DataFormats.FileDrop)).FirstOrDefault();
                HandleImageSelection(file);
            }
        }

        private void BtnBrowse_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog { Filter = "Image Files|*.jpg;*.jpeg;*.png" };
            if (openFileDialog.ShowDialog() == true) HandleImageSelection(openFileDialog.FileName);
        }

        private async void HandleImageSelection(string? filePath)
        {
            if (string.IsNullOrEmpty(filePath)) return;
            _lastInputPath = filePath; _lastOutputPath = null;
            _isEngineReady = false;
            
            ResetManualOffsets();

            BtnGenerate.IsEnabled = false;
            UpdateStatus("AI Processing: Landmark Mapping & Matting...", "#FFD700");
            ProcessingProgress.Visibility = Visibility.Visible;
            PlaceholderUI.Visibility = Visibility.Collapsed;
            PreviewImage.Visibility = Visibility.Collapsed;
            CompliancePanel.Visibility = Visibility.Collapsed;

            try 
            {
                await Task.Run(() => _processor.InitializeMasterFile(filePath));
                _isEngineReady = true;
                
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
                UpdateStatus($"Initialization Failed: {ex.Message}", "#FF5C5C");
                PlaceholderUI.Visibility = Visibility.Visible;
            }
            finally { ProcessingProgress.Visibility = Visibility.Collapsed; }
        }

        // ==========================================
        // SMART CANVAS INTERACTIONS
        // ==========================================
        private void Canvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (!_isEngineReady) return;
            _isDragging = true;
            _dragStartPos = e.GetPosition(CanvasBorder);
            CanvasBorder.CaptureMouse();
        }

        private void Canvas_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            _isDragging = false;
            CanvasBorder.ReleaseMouseCapture();
            TriggerLiveRender(); 
        }

        private void Canvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (!_isDragging || !_isEngineReady) return;

            Point currentPos = e.GetPosition(CanvasBorder);
            double deltaX = (currentPos.X - _dragStartPos.X) / 600.0 * 100.0;
            double deltaY = (currentPos.Y - _dragStartPos.Y) / 600.0 * 100.0;
            
            _panX += deltaX;
            _panY += deltaY;
            
            _dragStartPos = currentPos;
            TriggerLiveRender();
        }

        private void Canvas_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (!_isEngineReady) return;

            if (_isShiftDown) {
                // Tilt
                double tiltDelta = e.Delta > 0 ? 0.5 : -0.5;
                _tiltAngle = Math.Clamp(_tiltAngle + tiltDelta, -25, 25);
            } else {
                // Zoom anchored towards cursor
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

        private void ResetManualOffsets()
        {
            _panX = 0; _panY = 0; _zoomScale = 0; _tiltAngle = 0;
            if (_isEngineReady) TriggerLiveRender();
        }

        private void BtnSnapAuto_Click(object sender, RoutedEventArgs e) { ResetManualOffsets(); }

        private void Setting_Changed(object sender, RoutedEventArgs e) 
        {
            if (CanvasBorder != null && CmbBgColor != null)
                CanvasBorder.Background = (CmbBgColor.SelectedIndex == 0) ? new SolidColorBrush(Colors.White) : new SolidColorBrush(Color.FromRgb(242, 242, 242));
            TriggerLiveRender(); 
        }

        // ==========================================
        // RENDERING & COMPLIANCE HUD
        // ==========================================
        private async void TriggerLiveRender()
        {
            if (!_isEngineReady) return;
            _renderPending = true;
            if (_isRendering) return;

            _isRendering = true;
            try
            {
                while (_renderPending)
                {
                    _renderPending = false;
                    var req = GetCurrentRequest();
                    
                    // Render image
                    byte[] frame = await Task.Run(() => _processor.RenderFrame(req, previewMode: true));
                    ShowImageFromBytes(frame);
                    
                    // Compute and update compliance live
                    var report = _processor.ComputeCompliance(req);
                    UpdateComplianceUI(report);
                }
            }
            finally { _isRendering = false; }
        }

        private void UpdateComplianceUI(ComplianceReport rep)
        {
            Dispatcher.Invoke(() => {
                // Head Height
                ValHeadHeight.Text = rep.HeadHeight.ValueText;
                StatHeadHeight.Text = rep.HeadHeight.Pass ? "PASS" : "ADJUST";
                StatHeadHeight.Foreground = rep.HeadHeight.Pass ? new SolidColorBrush(Color.FromRgb(16, 185, 129)) : new SolidColorBrush(Color.FromRgb(245, 158, 11));

                // Eye Level
                ValEyeLevel.Text = rep.EyeLevel.ValueText;
                StatEyeLevel.Text = rep.EyeLevel.Pass ? "PASS" : "ADJUST";
                StatEyeLevel.Foreground = rep.EyeLevel.Pass ? new SolidColorBrush(Color.FromRgb(16, 185, 129)) : new SolidColorBrush(Color.FromRgb(245, 158, 11));

                // Centering
                ValCentering.Text = rep.Centering.ValueText;
                StatCentering.Text = rep.Centering.Pass ? "PASS" : "ADJUST";
                StatCentering.Foreground = rep.Centering.Pass ? new SolidColorBrush(Color.FromRgb(16, 185, 129)) : new SolidColorBrush(Color.FromRgb(245, 158, 11));

                // Head Level
                ValHeadLevel.Text = rep.HeadTilt.ValueText;
                StatHeadLevel.Text = rep.HeadTilt.Pass ? "PASS" : "ADJUST";
                StatHeadLevel.Foreground = rep.HeadTilt.Pass ? new SolidColorBrush(Color.FromRgb(16, 185, 129)) : new SolidColorBrush(Color.FromRgb(245, 158, 11));

                // Overall
                LblOverallPass.Text = rep.OverallPass ? "PASS" : "ADJUST";
                LblOverallPass.Foreground = rep.OverallPass ? new SolidColorBrush(Color.FromRgb(16, 185, 129)) : new SolidColorBrush(Color.FromRgb(245, 158, 11));
            });
        }

        private ProcessingRequest GetCurrentRequest()
        {
            PrintLayout layout = CmbLayout.SelectedIndex switch { 0 => PrintLayout.Print4x6, 1 => PrintLayout.Print4x8, 2 => PrintLayout.Print5x10, _ => PrintLayout.Digital2x2 };
            BackgroundColorOption bg = CmbBgColor.SelectedIndex == 0 ? BackgroundColorOption.White : BackgroundColorOption.OffWhite;
            EdgeFeatherMode edge = ChkSoftEdge.IsChecked == true ? EdgeFeatherMode.Soft : EdgeFeatherMode.Sharp;

            return new ProcessingRequest {
                Layout = layout, BgColor = bg, EdgeMode = edge,
                Tilt = _tiltAngle, ZoomOffset = _zoomScale,
                PanX = _panX, PanY = _panY,
                AutoFixLighting = ChkAutoLighting.IsChecked == true
            };
        }

        private async void BtnGenerate_Click(object sender, RoutedEventArgs e)
        {
            if (!_isEngineReady || string.IsNullOrEmpty(_lastInputPath)) return;
            BtnGenerate.IsEnabled = false;
            
            try
            {
                UpdateStatus("Compiling 600 DPI Master Print Sheet...", "#0066FF");
                ProcessingProgress.Visibility = Visibility.Visible;

                var req = GetCurrentRequest();
                string ext = "jpg";
                string outputFilename = $"{Path.GetFileNameWithoutExtension(_lastInputPath)}_VisaReady.{ext}";
                string outputPath = Path.Combine(Path.GetDirectoryName(_lastInputPath) ?? AppDomain.CurrentDomain.BaseDirectory, outputFilename);
                req.OutputPath = outputPath;

                await Task.Run(() => _processor.GenerateHighResPrint(req));
                _lastOutputPath = outputPath;
                
                UpdateStatus($"Generation Complete! Ready for Lab Printer.", "#00AA00");
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
                    bitmap.BeginInit();
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.StreamSource = stream;
                    bitmap.EndInit();
                }
                bitmap.Freeze();
                Dispatcher.Invoke(() => { PreviewImage.Source = bitmap; PreviewImage.Visibility = Visibility.Visible; });
            } catch {}
        }

        private void UpdateStatus(string message, string hexColor)
        {
            Dispatcher.Invoke(() => { LblStatus.Text = message; LblStatus.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString(hexColor)); });
        }
        
        private void BtnOpenFolder_Click(object sender, RoutedEventArgs e)
        {
            if (!string.IsNullOrEmpty(_lastOutputPath) && File.Exists(_lastOutputPath)) Process.Start("explorer.exe", $"/select,\"{_lastOutputPath}\"");
        }
    }
}
