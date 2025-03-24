using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Net.Http.Headers;
using System.Text.Json;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Storage;

namespace API_Use
{
    public partial class MainPage : ContentPage
    {
        // Observable collections for storing file paths
        public ObservableCollection<string> TrainingImages { get; set; } = new();
        public ObservableCollection<string> TestImages { get; set; } = new();
        public ObservableCollection<string> VerificationResults { get; set; } = new();

        public MainPage()
        {
            InitializeComponent();
            BindingContext = this;
            CleanImageDirectories();
            LoadTrainingImages();
            LoadTestImages();
        }

        // Helper method to convert a stream to a byte array
        public static byte[] ConvertStreamToBytes(Stream stream)
        {
            using MemoryStream memoryStream = new MemoryStream();
            stream.CopyTo(memoryStream);
            return memoryStream.ToArray();
        }

        // Cleans up previous image directories
        private void CleanImageDirectories()
        {
            string appDataDir = FileSystem.AppDataDirectory;
            string trainingDir = Path.Combine(appDataDir, "TrainingImages");
            string testDir = Path.Combine(appDataDir, "TestImages");

            try
            {
                if (Directory.Exists(trainingDir))
                    Directory.Delete(trainingDir, true);
                if (Directory.Exists(testDir))
                    Directory.Delete(testDir, true);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine("Error cleaning directories: " + ex.Message);
            }
        }

        // Loads training images from storage
        private void LoadTrainingImages()
        {
            string appDataDir = FileSystem.AppDataDirectory;
            string trainingDir = Path.Combine(appDataDir, "TrainingImages");
            if (Directory.Exists(trainingDir))
            {
                var files = Directory.GetFiles(trainingDir);
                TrainingImages.Clear();
                foreach (var file in files)
                    TrainingImages.Add(file);
            }
        }

        // Loads test images from storage
        private void LoadTestImages()
        {
            string appDataDir = FileSystem.AppDataDirectory;
            string testDir = Path.Combine(appDataDir, "TestImages");
            if (Directory.Exists(testDir))
            {
                var files = Directory.GetFiles(testDir);
                TestImages.Clear();
                foreach (var file in files)
                    TestImages.Add(file);
            }
        }

        // Event handler for capturing training selfies
        private async void OnCaptureTrainingSelfieClicked(object sender, EventArgs e)
        {
            string[] prompts = new[]
            {
                "Please move your face close to the camera and tap Capture.",
                "Now, please move your face a bit away and tap Capture.",
                "Finally, face straight towards the camera and tap Capture."
            };

            string appDataDir = FileSystem.AppDataDirectory;
            string trainingDir = Path.Combine(appDataDir, "TrainingImages");
            if (!Directory.Exists(trainingDir))
                Directory.CreateDirectory(trainingDir);

            foreach (var prompt in prompts)
            {
                bool shouldCapture = await DisplayAlert("Capture Selfie", prompt, "Capture", "Skip");
                if (shouldCapture)
                {
                    var photo = await MediaPicker.CapturePhotoAsync(new MediaPickerOptions { Title = "Selfie Capture" });
                    if (photo != null)
                    {
                        byte[] photoBytes;
                        using (var stream = await photo.OpenReadAsync())
                        {
                            photoBytes = ConvertStreamToBytes(stream);
                        }
                        string newFilePath = Path.Combine(trainingDir, $"{Guid.NewGuid()}.jpg");
                        await File.WriteAllBytesAsync(newFilePath, photoBytes);
                        TrainingImages.Add(newFilePath);
                    }
                }
            }
        }

        // Event handler for uploading test images
        private async void OnUploadTestImagesClicked(object sender, EventArgs e)
        {
            try
            {
                var results = await FilePicker.PickMultipleAsync(new PickOptions
                {
                    PickerTitle = "Select Test Images",
                    FileTypes = FilePickerFileType.Images
                });
                if (results != null)
                {
                    string appDataDir = FileSystem.AppDataDirectory;
                    string testDir = Path.Combine(appDataDir, "TestImages");
                    if (!Directory.Exists(testDir))
                        Directory.CreateDirectory(testDir);

                    foreach (var result in results)
                    {
                        byte[] imageBytes;
                        using (var stream = await result.OpenReadAsync())
                        {
                            imageBytes = ConvertStreamToBytes(stream);
                        }
                        string fileName = $"{Guid.NewGuid()}{Path.GetExtension(result.FileName)}";
                        string newFilePath = Path.Combine(testDir, fileName);
                        await File.WriteAllBytesAsync(newFilePath, imageBytes);
                        TestImages.Add(newFilePath);
                    }
                }
            }
            catch (Exception ex)
            {
                await DisplayAlert("Error", ex.Message, "OK");
            }
        }

        // Event handler for running verification
        private async void OnRunVerificationClicked(object sender, EventArgs e)
        {
            if (TrainingImages.Count == 0 || TestImages.Count == 0)
            {
                await DisplayAlert("Info", "Please capture at least one training image and upload at least one test image.", "OK");
                return;
            }

            VerificationResults.Clear();
            try
            {
                using var client = new HttpClient();

                foreach (var testImagePath in TestImages)
                {
                    using var form = new MultipartFormDataContent();

                    // Add all training images
                    foreach (var filePath in TrainingImages)
                    {
                        var trainingStream = File.OpenRead(filePath);
                        var streamContent = new StreamContent(trainingStream);
                        streamContent.Headers.ContentType = MediaTypeHeaderValue.Parse("image/jpeg");
                        form.Add(streamContent, "training_images", Path.GetFileName(filePath));
                    }

                    // Add the current test image
                    var testStream = File.OpenRead(testImagePath);
                    var testStreamContent = new StreamContent(testStream);
                    testStreamContent.Headers.ContentType = MediaTypeHeaderValue.Parse("image/jpeg");
                    form.Add(testStreamContent, "test_image", Path.GetFileName(testImagePath));

                    // Read the Haar Cascade XML directly from the file path in the app package
                    using var xmlStream = await FileSystem.OpenAppPackageFileAsync("haarcascade_frontalface_default.xml");
                    byte[] xmlBytes = ConvertStreamToBytes(xmlStream);
                    var xmlContent = new ByteArrayContent(xmlBytes);
                    xmlContent.Headers.ContentType = MediaTypeHeaderValue.Parse("application/xml");
                    form.Add(xmlContent, "haarcascade_xml", "haarcascade_frontalface_default.xml");

                    // Specify your FastAPI server URL
                    string serverUrl = "https://face-verification-fastapi.onrender.com/verify";
                    var response = await client.PostAsync(serverUrl, form);
                    response.EnsureSuccessStatusCode();

                    string responseString = await response.Content.ReadAsStringAsync();
                    bool verified = false;
                    using var doc = JsonDocument.Parse(responseString);
                    if (doc.RootElement.TryGetProperty("verified", out JsonElement verifiedElement))
                    {
                        verified = verifiedElement.GetBoolean();
                    }
                    string resultText = $"Test Image ({Path.GetFileName(testImagePath)}): " + (verified ? "Verified" : "Not Verified");
                    VerificationResults.Add(resultText);
                }
            }
            catch (Exception ex)
            {
                await DisplayAlert("Verification Error", ex.Message, "OK");
            }
        }
    }
}
