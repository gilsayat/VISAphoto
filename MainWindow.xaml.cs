using Microsoft.Win32;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using USVisaPhotoAutomator.Services;

namespace USVisaPhotoAutomator
{
    public partial class MainWindow : Window
    {
        private readonly VisaPhotoProcessor _processor;
        private string? _lastInputPath;
        private string? _lastOutputPath;

        public MainWindow()
        {
            InitializeComponent();
            string modelsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models");
            _processor = new VisaPhotoProcessor(modelsDir);
            Task.Run(() => { try { _processor.PreWarmModels(); UpdateStatus("Ready.", "#00FF7F"); } catch {} });
        }

        private void DropZone_DragEnter(object sender, DragEventArgs e) { if (e.Data.GetDataPresent(DataFormats.FileDrop)) e.Effects = DragDropEffects.Copy; }
        private void DropZone_DragLeave(object sender, DragEventArgs e) { }
        
        private void DropZone_Drop(object sender, DragEventArgs e)
        {
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

        private void HandleImageSelection(string? filePath)
        {
            if (string.IsNullOrEmpty(filePath)) return;
            _lastInputPath = filePath; _lastOutputPath = null;
            
            // Reset sliders
            ZoomSlider.Value = 0; TiltSlider.Value = 0; PanXSlider.Value = 0; PanYSlider.Value = 0;
            BrightSlider.Value = 0; TempSlider.Value = 0; ChokeSlider.Value = 0;

            ShowImageInPreview(filePath);
            BtnGenerate.IsEnabled = true;
            BtnOpenFolder.Visibility = Visibility.Collapsed;
            VisaHUD.Visibility = Visibility.Visible;
            UpdateStatus("Adjust framing in the viewport, then Generate.", "#00FF7F");
        }

        private void Slider_UpdateUI(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (PreviewRotation == null || PreviewScale == null || PreviewTranslate == null) return;
            
            PreviewRotation.Angle = TiltSlider.Value;
            double scale = 1.0 + (ZoomSlider.Value / 100.0);
            PreviewScale.ScaleX = scale; PreviewScale.ScaleY = scale;
            PreviewTranslate.X = PanXSlider.Value;
            PreviewTranslate.Y = PanYSlider.Value;

            if (LblZ != null) {
                LblZ.Text = $"{ZoomSlider.Value:F0}"; LblT.Text = $"{TiltSlider.Value:F1}°";
                LblX.Text = $"{PanXSlider.Value:F0}"; LblY.Text = $"{PanYSlider.Value:F0}";
                LblB.Text = $"{BrightSlider.Value:F0}"; LblC.Text = $"{TempSlider.Value:F0}";
                LblE.Text = $"{ChokeSlider.Value:F0}";
            }
        }

        private async void BtnGenerate_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_lastInputPath) || !File.Exists(_lastInputPath)) return;
            BtnGenerate.IsEnabled = false;
            
            try
            {
                UpdateStatus("Processing AI Segments & Matrices...", "#FFD700");
                ProcessingProgress.Visibility = Visibility.Visible; VisaHUD.Visibility = Visibility.Collapsed;

                string layoutTag = (CmbLayout.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? "Digital2x2";
                PrintLayout layout = layoutTag switch { "Print4x6" => PrintLayout.Print4x6, "Print4x8" => PrintLayout.Print4x8, "Print5x10" => PrintLayout.Print5x10, _ => PrintLayout.Digital2x2 };
                string bgTag = (CmbBgColor.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "Pure White";
                BackgroundColorOption bgColor = bgTag == "Off-White" ? BackgroundColorOption.OffWhite : BackgroundColorOption.White;
                string edgeTag = (CmbEdgeMode.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? "AutoSmart";
                EdgeFeatherMode edgeMode = edgeTag switch { "Sharp" => EdgeFeatherMode.Sharp, "Soft" => EdgeFeatherMode.Soft, _ => EdgeFeatherMode.AutoSmart };

                string outputFilename = $"{Path.GetFileNameWithoutExtension(_lastInputPath)}_Visa.jpg";
                string outputPath = Path.Combine(Path.GetDirectoryName(_lastInputPath) ?? AppDomain.CurrentDomain.BaseDirectory, outputFilename);

                var req = new ProcessingRequest {
                    InputPath = _lastInputPath, OutputPath = outputPath,
                    Layout = layout, BgColor = bgColor, EdgeMode = edgeMode,
                    Tilt = TiltSlider.Value, ZoomOffset = ZoomSlider.Value,
                    PanX = PanXSlider.Value, PanY = PanYSlider.Value,
                    Brightness = BrightSlider.Value, ColorTemp = TempSlider.Value, EdgeChoke = (int)ChokeSlider.Value
                };

                await Task.Run(() => _processor.ProcessVisaPhoto(req));
                _lastOutputPath = outputPath;
                ShowImageInPreview(outputPath);
                UpdateStatus($"Successfully generated studio file!", "#00FF7F");
                BtnOpenFolder.Visibility = Visibility.Visible;
            }
            catch (Exception ex) { UpdateStatus($"Error: {ex.Message}", "#FF5C5C"); }
            finally { ProcessingProgress.Visibility = Visibility.Collapsed; BtnGenerate.IsEnabled = true; }
        }

        private void ShowImageInPreview(string path)
        {
            BitmapImage bitmap = new BitmapImage();
            bitmap.BeginInit(); bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.CreateOptions = BitmapCreateOptions.IgnoreImageCache;
            bitmap.UriSource = new Uri(path); bitmap.EndInit();
            PlaceholderUI.Visibility = Visibility.Collapsed; PreviewImage.Source = bitmap; PreviewImage.Visibility = Visibility.Visible;
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
