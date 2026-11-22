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
        private bool _isProcessing;

        public MainWindow()
        {
            InitializeComponent();
            string modelsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models");
            _processor = new VisaPhotoProcessor(modelsDir);

            Task.Run(() =>
            {
                try
                {
                    UpdateStatus("Warming up AI Models...", "#FFD700");
                    _processor.PreWarmModels();
                    UpdateStatus("Ready for input.", "#00FF7F");
                }
                catch (Exception ex)
                {
                    UpdateStatus($"Initialization Error: {ex.Message}", "#FF5C5C");
                }
            });
        }

        private void DropZone_DragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                e.Effects = DragDropEffects.Copy;
                DropBorder.BorderBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1E90FF"));
            }
        }

        private void DropZone_DragLeave(object sender, DragEventArgs e)
        {
            DropBorder.BorderBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#3F3F46"));
        }

        private void DropZone_Drop(object sender, DragEventArgs e)
        {
            DropBorder.BorderBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#3F3F46"));
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                string[]? files = e.Data.GetData(DataFormats.FileDrop) as string[];
                string? file = files?.FirstOrDefault();
                HandleImageSelection(file);
            }
        }

        private void BtnBrowse_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Title = "Select a Portrait Photo",
                Filter = "Image Files|*.jpg;*.jpeg;*.png"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                HandleImageSelection(openFileDialog.FileName);
            }
        }

        private void HandleImageSelection(string? filePath)
        {
            if (string.IsNullOrEmpty(filePath) || !IsImageFile(filePath))
            {
                UpdateStatus("Error: Invalid image file.", "#FF5C5C");
                return;
            }

            _lastInputPath = filePath;
            _lastOutputPath = null;

            TiltSlider.Value = 0;
            ZoomSlider.Value = 56;
            if (PreviewRotation != null) PreviewRotation.Angle = 0;
            if (PreviewScale != null) { PreviewScale.ScaleX = 1.0; PreviewScale.ScaleY = 1.0; }

            ShowImageInPreview(filePath);
            BtnGenerate.IsEnabled = true;
            BtnOpenFolder.Visibility = Visibility.Collapsed;
            AlignmentGrid.Visibility = ChkGrid.IsChecked == true ? Visibility.Visible : Visibility.Collapsed;
            UpdateStatus("Adjust tilt & zoom, then click Generate.", "#00FF7F");
        }

        private void ChkGrid_Click(object sender, RoutedEventArgs e)
        {
            if (AlignmentGrid != null && !string.IsNullOrEmpty(_lastInputPath))
            {
                AlignmentGrid.Visibility = ChkGrid.IsChecked == true ? Visibility.Visible : Visibility.Collapsed;
            }
        }

        private void TiltSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (PreviewRotation != null)
            {
                PreviewRotation.Angle = e.NewValue;
                if (LblTiltAngle != null) LblTiltAngle.Text = $"{e.NewValue:F1}°";
            }
        }
        
        private void ZoomSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (PreviewScale != null)
            {
                double scale = e.NewValue / 56.0;
                PreviewScale.ScaleX = scale;
                PreviewScale.ScaleY = scale;
                if (LblZoomValue != null) LblZoomValue.Text = $"{e.NewValue:F0}%";
            }
        }

        private async void BtnGenerate_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_lastInputPath) || !File.Exists(_lastInputPath)) return;
            
            BtnGenerate.IsEnabled = false;
            await ProcessFileAsync(_lastInputPath);
            BtnGenerate.IsEnabled = true;
        }

        private bool IsImageFile(string path)
        {
            string ext = Path.GetExtension(path).ToLowerInvariant();
            return ext == ".jpg" || ext == ".jpeg" || ext == ".png";
        }

        private async Task ProcessFileAsync(string inputPath)
        {
            if (_isProcessing) return;
            _isProcessing = true;

            try
            {
                UpdateStatus("Processing AI Image & Rendering Sheet...", "#FFD700");
                ProcessingProgress.Visibility = Visibility.Visible;
                BtnOpenFolder.Visibility = Visibility.Collapsed;
                AlignmentGrid.Visibility = Visibility.Collapsed; // Hide grid during render

                string format = (CmbFormat.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "JPG";
                bool fixLighting = ChkFixLighting.IsChecked == true;
                double manualTilt = TiltSlider.Value;
                double headZoomRatio = ZoomSlider.Value / 100.0;
                
                string layoutTag = (CmbLayout.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? "Digital2x2";
                PrintLayout layout = layoutTag switch {
                    "Print4x6" => PrintLayout.Print4x6,
                    "Print4x8" => PrintLayout.Print4x8,
                    "Print5x10" => PrintLayout.Print5x10,
                    _ => PrintLayout.Digital2x2
                };
                
                int dpi = (CmbDPI.SelectedIndex == 1) ? 600 : 300;
                
                string bgTag = (CmbBgColor.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "White";
                BackgroundColorOption bgColor = bgTag == "Off-White" ? BackgroundColorOption.OffWhite : BackgroundColorOption.White;

                string edgeTag = (CmbEdgeMode.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? "AutoSmart";
                EdgeFeatherMode edgeMode = edgeTag switch
                {
                    "Sharp" => EdgeFeatherMode.Sharp,
                    "Balanced" => EdgeFeatherMode.Balanced,
                    "Soft" => EdgeFeatherMode.Soft,
                    _ => EdgeFeatherMode.AutoSmart
                };

                string ext = format.ToLowerInvariant();
                string outputFilename = $"{Path.GetFileNameWithoutExtension(inputPath)}_VisaCompliant.{ext}";
                string outputDir = Path.GetDirectoryName(inputPath) ?? AppDomain.CurrentDomain.BaseDirectory;
                string outputPath = Path.Combine(outputDir, outputFilename);

                await Task.Run(() => _processor.ProcessVisaPhoto(
                    inputPath, outputPath, format, layout, dpi, bgColor, edgeMode, fixLighting, manualTilt, headZoomRatio));

                _lastOutputPath = outputPath;
                
                if (PreviewRotation != null) PreviewRotation.Angle = 0;
                if (PreviewScale != null) { PreviewScale.ScaleX = 1.0; PreviewScale.ScaleY = 1.0; }
                ShowImageInPreview(outputPath);
                
                UpdateStatus($"Saved {layout} @ {dpi} DPI ({edgeMode})!", "#00FF7F");
                BtnOpenFolder.Visibility = Visibility.Visible;
            }
            catch (Exception ex)
            {
                UpdateStatus($"Error: {ex.Message}", "#FF5C5C");
            }
            finally
            {
                ProcessingProgress.Visibility = Visibility.Collapsed;
                _isProcessing = false;
            }
        }

        private void ShowImageInPreview(string path)
        {
            BitmapImage bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.CreateOptions = BitmapCreateOptions.IgnoreImageCache;
            bitmap.UriSource = new Uri(path);
            bitmap.EndInit();
            
            PlaceholderUI.Visibility = Visibility.Collapsed;
            PreviewImage.Source = bitmap;
            PreviewImage.Visibility = Visibility.Visible;
        }

        private void UpdateStatus(string message, string hexColor)
        {
            Dispatcher.Invoke(() =>
            {
                LblStatus.Text = message;
                LblStatus.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString(hexColor));
            });
        }

        private void BtnOpenFolder_Click(object sender, RoutedEventArgs e)
        {
            if (!string.IsNullOrEmpty(_lastOutputPath) && File.Exists(_lastOutputPath))
            {
                Process.Start("explorer.exe", $"/select,\"{_lastOutputPath}\"");
            }
        }

        protected override void OnClosed(EventArgs e)
        {
            _processor.Dispose();
            base.OnClosed(e);
        }
    }
}
