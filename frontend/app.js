const imageInput = document.getElementById('imageInput');
const processBtn = document.getElementById('processBtn');
const loader = document.getElementById('loader');
const results = document.getElementById('results');
const originalImg = document.getElementById('originalImg');
const enhancedImg = document.getElementById('enhancedImg');
const downloadBtn = document.getElementById('downloadBtn');
const processingTime = document.getElementById('processingTime');

let selectedFile = null;

// File validation constants
const MAX_FILE_SIZE_MB = 10;
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg', 'image/bmp', 'image/tiff'];

function validateFile(file) {
    if (!file) {
        return { valid: false, message: 'No file selected' };
    }

    // Check file type
    if (!ALLOWED_TYPES.includes(file.type)) {
        return {
            valid: false,
            message: 'Invalid file type. Please upload JPG, PNG, BMP, or TIFF images.'
        };
    }

    // Check file size
    const fileSizeMB = file.size / (1024 * 1024);
    if (fileSizeMB > MAX_FILE_SIZE_MB) {
        return {
            valid: false,
            message: `File too large (${fileSizeMB.toFixed(1)}MB). Maximum size is ${MAX_FILE_SIZE_MB}MB.`
        };
    }

    return { valid: true };
}

function showError(message) {
    alert('❌ ' + message);
}

function updateLoader(message) {
    const loaderText = loader.querySelector('p');
    if (loaderText) {
        loaderText.textContent = message;
    }
}

imageInput.addEventListener('change', (e) => {
    selectedFile = e.target.files[0];

    if (selectedFile) {
        // Validate file
        const validation = validateFile(selectedFile);
        if (!validation.valid) {
            showError(validation.message);
            selectedFile = null;
            processBtn.disabled = true;
            return;
        }

        processBtn.disabled = false;

        // Preview original image
        const reader = new FileReader();
        reader.onload = (event) => {
            originalImg.src = event.target.result;
            results.classList.remove('hidden');
            enhancedImg.src = ''; // Clear previous
            if (processingTime) processingTime.textContent = '';
        };
        reader.readAsDataURL(selectedFile);
    }
});

processBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Double-check validation
    const validation = validateFile(selectedFile);
    if (!validation.valid) {
        showError(validation.message);
        return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    processBtn.disabled = true;
    loader.classList.remove('hidden');
    updateLoader('Estimating depth...');

    const startTime = Date.now();

    try {
        updateLoader('Processing with Sea-Thru algorithm...');

        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Processing failed');
        }

        const data = await response.json();

        // Calculate client-side processing time
        const clientTime = ((Date.now() - startTime) / 1000).toFixed(2);

        // Display enhanced image
        enhancedImg.src = data.enhanced;

        // Set download link with meaningful filename
        const originalName = selectedFile.name.split('.')[0];
        downloadBtn.href = data.enhanced;
        downloadBtn.download = `enhanced_${originalName}.png`;

        // Show processing time
        if (processingTime) {
            processingTime.textContent = `Processing completed in ${data.processing_time || clientTime}s`;
        }

        loader.classList.add('hidden');
        updateLoader('Restoring colors...');

        console.log('Processing successful:', data);
    } catch (error) {
        console.error('Processing error:', error);
        showError(`Processing failed: ${error.message}`);
        loader.classList.add('hidden');
        processBtn.disabled = false;
    }
});

// Enable processing button again when a new image is selected
imageInput.addEventListener('change', () => {
    if (enhancedImg.src) {
        processBtn.disabled = false;
    }
});
