using Microsoft.Win32;
using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
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

        public MainWindow()
        {
            InitializeComponent();
            string modelsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models");
            _processor = new VisaPhotoProcessor(modelsDir);
            Task.Run(() => { try { _processor.PreWarmModels(); UpdateStatus("Engine ready.", "#00FF7F"); } catch {} });
        }

        private void DropZone_DragEnter(object sender, DragEventArgs e) { if (e.Data.GetDataPresent(DataFormats.FileDrop)) { e.Effects = DragDropEffects.Copy; DropBorder.BorderBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1E90FF")); } }
        private void DropZone_DragLeave(object sender, DragEventArgs e) { DropBorder.BorderBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#333338")); }
        
        private void DropZone_Drop(object sender, DragEventArgs e)
        {
            DropBorder.BorderBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#333338"));
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                var file = (e.Data.GetData(DataFormats.FileDrop) as string[])?.FirstOrDefault();
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
            
            // Reset Sliders silently
            ZoomSlider.Value = 0; TiltSlider.Value = 0; PanXSlider.Value = 0; PanYSlider.Value = 0;
            BrightSlider.Value = 0; RecovSlider.Value = 0; TempSlider.Value = 0; ChokeSlider.Value = 0;

            BtnGenerate.IsEnabled = false;
            UpdateStatus("Extracting Master Alpha Mask & Landmarks...", "#FFD700");
            ProcessingProgress.Visibility = Visibility.Visible;
            PlaceholderUI.Visibility = Visibility.Collapsed;
            PreviewImage.Visibility = Visibility.Collapsed;

            try 
            {
                await Task.Run(() => _processor.InitializeMasterFile(filePath));
                _isEngineReady = true;
                VisaHUD.Visibility = Visibility.Visible;
                BtnGenerate.IsEnabled = true;
                BtnOpenFolder.Visibility = Visibility.Collapsed;
                UpdateStatus("WYSIWYG Mode Active. Adjust sliders.", "#00FF7F");
                
                TriggerLiveRender();
            } 
            catch (Exception ex) 
            {
                UpdateStatus($"Initialization Failed: {ex.Message}", "#FF5C5C");
                PlaceholderUI.Visibility = Visibility.Visible;
            }
            finally { ProcessingProgress.Visibility = Visibility.Collapsed; }
        }

        private void Slider_UpdateUI(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (LblZ != null) {
                LblZ.Text = $"{ZoomSlider.Value:F0}"; LblT.Text = $"{TiltSlider.Value:F1}°";
                LblX.Text = $"{PanXSlider.Value:F0}"; LblY.Text = $"{PanYSlider.Value:F0}";
                LblB.Text = $"{BrightSlider.Value:F0}"; LblC.Text = $"{TempSlider.Value:F0}";
                LblE.Text = $"{ChokeSlider.Value:F0}"; LblR.Text = $"{RecovSlider.Value:F0}";
            }
            TriggerLiveRender();
        }

        private void Cmb_SelectionChanged(object sender, SelectionChangedEventArgs e) { TriggerLiveRender(); }

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
                    
                    // Render exactly what the final output will be (but at a fast preview resolution of 600x600)
                    byte[] frame = await Task.Run(() => _processor.RenderFrame(req, previewMode: true));
                    ShowImageFromBytes(frame);
                }
            }
            finally
            {
                _isRendering = false;
            }
        }

        private ProcessingRequest GetCurrentRequest()
        {
            string layoutTag = (CmbLayout.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? "Digital2x2";
            PrintLayout layout = layoutTag switch { "Print4x6" => PrintLayout.Print4x6, "Print4x8" => PrintLayout.Print4x8, "Print5x10" => PrintLayout.Print5x10, _ => PrintLayout.Digital2x2 };
            string bgTag = (CmbBgColor.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "Pure White";
            BackgroundColorOption bgColor = bgTag == "Off-White" ? BackgroundColorOption.OffWhite : BackgroundColorOption.White;
            string edgeTag = (CmbEdgeMode.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? "AutoSmart";
            EdgeFeatherMode edgeMode = edgeTag switch { "Sharp" => EdgeFeatherMode.Sharp, "Soft" => EdgeFeatherMode.Soft, _ => EdgeFeatherMode.AutoSmart };

            return new ProcessingRequest {
                Layout = layout, BgColor = bgColor, EdgeMode = edgeMode,
                Tilt = TiltSlider.Value, ZoomOffset = ZoomSlider.Value,
                PanX = PanXSlider.Value, PanY = PanYSlider.Value,
                Brightness = BrightSlider.Value, ColorTemp = TempSlider.Value, 
                HighlightRecovery = RecovSlider.Value, EdgeChoke = (int)ChokeSlider.Value
            };
        }

        private async void BtnGenerate_Click(object sender, RoutedEventArgs e)
        {
            if (!_isEngineReady || string.IsNullOrEmpty(_lastInputPath)) return;
            BtnGenerate.IsEnabled = false;
            
            try
            {
                UpdateStatus("Compiling 600 DPI Print Sheet...", "#FFD700");
                ProcessingProgress.Visibility = Visibility.Visible;

                var req = GetCurrentRequest();
                string ext = (req.Layout == PrintLayout.Digital2x2) ? "jpg" : "jpg"; // Use jpg for prints
                string outputFilename = $"{Path.GetFileNameWithoutExtension(_lastInputPath)}_VisaReady.{ext}";
                string outputPath = Path.Combine(Path.GetDirectoryName(_lastInputPath) ?? AppDomain.CurrentDomain.BaseDirectory, outputFilename);
                req.OutputPath = outputPath;

                // Render at full 600 DPI (1200x1200px master)
                await Task.Run(() => _processor.GenerateHighResPrint(req));
                _lastOutputPath = outputPath;
                
                UpdateStatus($"Successfully generated print file!", "#00FF7F");
                BtnOpenFolder.Visibility = Visibility.Visible;
            }
            catch (Exception ex) { UpdateStatus($"Export Error: {ex.Message}", "#FF5C5C"); }
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
                Dispatcher.Invoke(() => {
                    PreviewImage.Source = bitmap; 
                    PreviewImage.Visibility = Visibility.Visible;
                });
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
