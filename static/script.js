document.getElementById("upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();

    const fileInput = document.getElementById("file-input");
    if (!fileInput.files[0]) {
        alert("Please select an image.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // Send file to the backend
    const response = await fetch("/predict/", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        alert("Error uploading file. Please try again.");
        return;
    }

    const data = await response.json();

    // Display predictions
    const predictionsList = document.getElementById("predictions");
    predictionsList.innerHTML = `
        <li><strong>Baseline CNN:</strong> ${data["Baseline CNN Prediction"]}</li>
        <li><strong>VGG16:</strong> ${data["VGG16 Prediction"]}</li>
        <li><strong>Random Forest:</strong> ${data["Random Forest Prediction"]}</li>
    `;

    // Display Grad-CAM
    const gradcamImage = document.getElementById("gradcam-image");
    gradcamImage.src = "/" + data["Grad-CAM"]; // 确保路径正确

    // Show results section
    document.getElementById("results").classList.remove("hidden");
});