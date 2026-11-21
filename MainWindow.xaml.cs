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

        private async void DropZone_Drop(object sender, DragEventArgs e)
        {
            DropBorder.BorderBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#3F3F46"));
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                string[]? files = e.Data.GetData(DataFormats.FileDrop) as string[];
                string? file = files?.FirstOrDefault();
                await HandleImageSelectionAsync(file);
            }
        }

        private async void BtnBrowse_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Title = "Select a Portrait Photo",
                Filter = "Image Files|*.jpg;*.jpeg;*.png"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                await HandleImageSelectionAsync(openFileDialog.FileName);
            }
        }

        private async void Settings_Changed(object sender, RoutedEventArgs e)
        {
            if (!IsLoaded || _isProcessing || string.IsNullOrEmpty(_lastInputPath) || !File.Exists(_lastInputPath))
                return;

            await ProcessFileAsync(_lastInputPath);
        }

        private async Task HandleImageSelectionAsync(string? filePath)
        {
            if (string.IsNullOrEmpty(filePath) || !IsImageFile(filePath))
            {
                UpdateStatus("Error: Invalid image file.", "#FF5C5C");
                return;
            }

            _lastInputPath = filePath;
            await ProcessFileAsync(filePath);
        }

        private bool IsImageFile(string path)
        {
            string ext = Path.GetExtension(path).ToLowerInvariant();
            return ext == ".jpg" || ext == ".jpeg" || ext == ".png";
        }

        private EdgeFeatherMode GetSelectedEdgeMode()
        {
            string? tag = (CmbEdgeMode?.SelectedItem as ComboBoxItem)?.Tag?.ToString();
            return tag switch
            {
                "Sharp" => EdgeFeatherMode.Sharp,
                "Balanced" => EdgeFeatherMode.Balanced,
                "Soft" => EdgeFeatherMode.Soft,
                _ => EdgeFeatherMode.AutoSmart
            };
        }

        private async Task ProcessFileAsync(string inputPath)
        {
            if (_isProcessing) return;
            _isProcessing = true;

            try
            {
                UpdateStatus("Processing AI Image...", "#FFD700");
                PlaceholderUI.Visibility = Visibility.Collapsed;
                PreviewImage.Visibility = Visibility.Collapsed;
                BtnOpenFolder.Visibility = Visibility.Collapsed;
                ProcessingProgress.Visibility = Visibility.Visible;

                string format = (CmbFormat.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "JPG";
                bool isPrintMode = ChkPrintMode.IsChecked == true;
                EdgeFeatherMode edgeMode = GetSelectedEdgeMode();

                string ext = format.ToLowerInvariant();
                string outputFilename = $"{Path.GetFileNameWithoutExtension(inputPath)}_VisaCompliant.{ext}";
                string outputDir = Path.GetDirectoryName(inputPath) ?? AppDomain.CurrentDomain.BaseDirectory;
                string outputPath = Path.Combine(outputDir, outputFilename);
                _lastOutputPath = outputPath;

                await Task.Run(() => _processor.ProcessVisaPhoto(inputPath, outputPath, format, isPrintMode, edgeMode));

                ShowPreview(outputPath);
                UpdateStatus($"Compliant & Saved ({edgeMode})!", "#00FF7F");
                BtnOpenFolder.Visibility = Visibility.Visible;
            }
            catch (Exception ex)
            {
                UpdateStatus($"Error: {ex.Message}", "#FF5C5C");
                PlaceholderUI.Visibility = Visibility.Visible;
            }
            finally
            {
                ProcessingProgress.Visibility = Visibility.Collapsed;
                _isProcessing = false;
            }
        }

        private void ShowPreview(string path)
        {
            BitmapImage bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.CreateOptions = BitmapCreateOptions.IgnoreImageCache;
            bitmap.UriSource = new Uri(path);
            bitmap.EndInit();
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
