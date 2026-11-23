/**
 * AI-Based Anemia Detection - Interactive Client Script
 * Handles image upload preview, modality-model dependency synchronization,
 * and analysis progress states.
 */

document.addEventListener("DOMContentLoaded", function () {
  // 1. Modality & Model Dynamic Dependency
  const modalitySelect = document.getElementById("modality-select");
  const modelSelect = document.getElementById("model-select");
  const modelTypeBadge = document.getElementById("model-type-badge");

  const modelOptions = {
    eye_conjunctiva: [
      { id: "resnet18", name: "Pretrained ResNet-18 (Deep Learning)", type: "Deep Learning" },
      { id: "custom_cnn", name: "Custom 4-Block CNN (Deep Learning)", type: "Deep Learning" },
      { id: "svm", name: "Support Vector Machine - RBF (Classical ML)", type: "Classical ML" },
      { id: "random_forest", name: "Random Forest (Classical ML)", type: "Classical ML" },
      { id: "logistic_regression", name: "Logistic Regression (Classical ML)", type: "Classical ML" },
      { id: "knn", name: "K-Nearest Neighbors - k=15 (Classical ML)", type: "Classical ML" },
      { id: "decision_tree", name: "Decision Tree (Classical ML)", type: "Classical ML" }
    ],
    palmar_pallor: [
      { id: "svm", name: "Support Vector Machine - RBF (Classical ML)", type: "Classical ML" },
      { id: "random_forest", name: "Random Forest (Classical ML)", type: "Classical ML" },
      { id: "logistic_regression", name: "Logistic Regression (Classical ML)", type: "Classical ML" },
      { id: "knn", name: "K-Nearest Neighbors - k=15 (Classical ML)", type: "Classical ML" },
      { id: "decision_tree", name: "Decision Tree (Classical ML)", type: "Classical ML" }
    ],
    nail_beds: [
      { id: "svm", name: "Support Vector Machine - RBF (Classical ML)", type: "Classical ML" },
      { id: "random_forest", name: "Random Forest (Classical ML)", type: "Classical ML" },
      { id: "logistic_regression", name: "Logistic Regression (Classical ML)", type: "Classical ML" },
      { id: "knn", name: "K-Nearest Neighbors - k=15 (Classical ML)", type: "Classical ML" },
      { id: "decision_tree", name: "Decision Tree (Classical ML)", type: "Classical ML" }
    ]
  };

  function updateModelOptions() {
    if (!modalitySelect || !modelSelect) return;
    const selectedModality = modalitySelect.value;
    const available = modelOptions[selectedModality] || [];

    modelSelect.innerHTML = "";
    available.forEach((opt) => {
      const optionElem = document.createElement("option");
      optionElem.value = opt.id;
      optionElem.textContent = opt.name;
      optionElem.dataset.type = opt.type;
      modelSelect.appendChild(optionElem);
    });

    updateBadge();
  }

  function updateBadge() {
    if (!modelSelect || !modelTypeBadge) return;
    const selected = modelSelect.options[modelSelect.selectedIndex];
    if (selected && selected.dataset.type) {
      modelTypeBadge.textContent = selected.dataset.type;
      modelTypeBadge.className = selected.dataset.type === "Deep Learning" ? "badge badge-primary" : "badge badge-secondary";
    }
  }

  if (modalitySelect && modelSelect) {
    modalitySelect.addEventListener("change", updateModelOptions);
    modelSelect.addEventListener("change", updateBadge);
    updateModelOptions();
  }

  // 2. Drag & Drop Image Upload with Instant Preview
  const dropzone = document.getElementById("upload-dropzone");
  const fileInput = document.getElementById("file-input");
  const previewContainer = document.getElementById("preview-container");
  const previewImage = document.getElementById("preview-image");
  const removeBtn = document.getElementById("remove-btn");
  const fileNameDisplay = document.getElementById("file-name-display");

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect(fileInput.files[0]);
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files.length > 0) {
        handleFileSelect(fileInput.files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    if (!file) return;

    // Validate extension
    const validExts = ["image/jpeg", "image/png", "image/jpg"];
    if (!validExts.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png)$/i)) {
      alert("Invalid format! Please upload a valid JPG, JPEG, or PNG image.");
      clearFile();
      return;
    }

    // Validate size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
      alert("File is too large! Maximum allowed upload size is 16 MB.");
      clearFile();
      return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
      if (previewImage) previewImage.src = e.target.result;
      if (previewContainer) previewContainer.style.display = "block";
      if (dropzone) dropzone.style.display = "none";
      if (fileNameDisplay) fileNameDisplay.textContent = file.name + ` (${(file.size / 1024).toFixed(1)} KB)`;
    };
    reader.readAsDataURL(file);
  }

  function clearFile() {
    if (fileInput) fileInput.value = "";
    if (previewImage) previewImage.src = "";
    if (previewContainer) previewContainer.style.display = "none";
    if (dropzone) dropzone.style.display = "block";
    if (fileNameDisplay) fileNameDisplay.textContent = "";
  }

  if (removeBtn) {
    removeBtn.addEventListener("click", (e) => {
      e.preventDefault();
      clearFile();
    });
  }

  // 3. Form Submission Loading State
  const analysisForm = document.getElementById("analysis-form");
  const submitBtn = document.getElementById("submit-btn");
  const submitBtnText = document.getElementById("submit-btn-text");
  const submitSpinner = document.getElementById("submit-spinner");

  if (analysisForm && submitBtn) {
    analysisForm.addEventListener("submit", function (e) {
      if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        e.preventDefault();
        alert("Please select or drop an image before initiating analysis.");
        return;
      }
      submitBtn.disabled = true;
      if (submitBtnText) submitBtnText.textContent = "Analyzing & Generating Explanations...";
      if (submitSpinner) submitSpinner.style.display = "inline-block";
    });
  }
});
