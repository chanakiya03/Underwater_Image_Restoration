// Tab Switcher Logic
const p1Tab = document.getElementById('p1Tab');
const p2Tab = document.getElementById('p2Tab');
const p1Section = document.getElementById('p1Section');
const p2Section = document.getElementById('p2Section');

p1Tab.addEventListener('click', () => {
    p1Tab.classList.add('active');
    p2Tab.classList.remove('active');
    p1Section.classList.remove('hidden');
    p2Section.classList.add('hidden');
});

p2Tab.addEventListener('click', () => {
    p2Tab.classList.add('active');
    p1Tab.classList.remove('active');
    p2Section.classList.remove('hidden');
    p1Section.classList.add('hidden');
});

// Pipeline 1: Restoration
const imageInput = document.getElementById('imageInput');
const processBtn = document.getElementById('processBtn');
const loader = document.getElementById('loader');
const results = document.getElementById('results');
const originalImg = document.getElementById('originalImg');
const enhancedImg = document.getElementById('enhancedImg');
const downloadBtn = document.getElementById('downloadBtn');
const processingTime = document.getElementById('processingTime');

let selectedFile = null;

// Pipeline 2: Mosaicking
const mosaicInput = document.getElementById('mosaicInput');
const mosaicBtn = document.getElementById('mosaicBtn');
const mosaicLoader = document.getElementById('mosaicLoader');
const mosaicResults = document.getElementById('mosaicResults');
const mosaicImg = document.getElementById('mosaicImg');
const matchVisImg = document.getElementById('matchVisImg');
const mosaicProcessingTime = document.getElementById('mosaicProcessingTime');
const mosaicDownloadBtn = document.getElementById('mosaicDownloadBtn');

let selectedFiles = [];

// File validation constants
const MAX_FILE_SIZE_MB = 10;
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg', 'image/bmp', 'image/tiff'];

function validateFile(file) {
    if (!file) {
        return { valid: false, message: 'No file selected' };
    }
    if (!ALLOWED_TYPES.includes(file.type)) {
        return { valid: false, message: 'Invalid file type. Please upload JPG, PNG, BMP, or TIFF images.' };
    }
    const fileSizeMB = file.size / (1024 * 1024);
    if (fileSizeMB > MAX_FILE_SIZE_MB) {
        return { valid: false, message: `File too large (${fileSizeMB.toFixed(1)}MB). Max size is ${MAX_FILE_SIZE_MB}MB.` };
    }
    return { valid: true };
}

function showError(message) {
    alert('❌ ' + message);
}

function updateLoader(loaderElement, message) {
    const loaderText = loaderElement.querySelector('p');
    if (loaderText) {
        loaderText.textContent = message;
    }
}

// P1 Events
imageInput.addEventListener('change', (e) => {
    selectedFile = e.target.files[0];
    if (selectedFile) {
        const validation = validateFile(selectedFile);
        if (!validation.valid) {
            showError(validation.message);
            selectedFile = null;
            processBtn.disabled = true;
            return;
        }
        processBtn.disabled = false;
        const reader = new FileReader();
        reader.onload = (event) => {
            originalImg.src = event.target.result;
            results.classList.remove('hidden');
            enhancedImg.src = '';
            if (processingTime) processingTime.textContent = '';
        };
        reader.readAsDataURL(selectedFile);
    }
});

processBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append('file', selectedFile);

    processBtn.disabled = true;
    loader.classList.remove('hidden');
    updateLoader(loader, 'Estimating depth...');

    try {
        updateLoader(loader, 'Applying Sea-Thru algorithm...');
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Processing failed');
        }

        const data = await response.json();
        enhancedImg.src = data.enhanced;
        downloadBtn.href = data.enhanced;
        if (processingTime) {
            processingTime.textContent = `Restoration completed in ${data.processing_time}s`;
        }
        loader.classList.add('hidden');
    } catch (error) {
        console.error(error);
        showError(error.message);
        loader.classList.add('hidden');
        processBtn.disabled = false;
    }
});

// P2 Events
mosaicInput.addEventListener('change', (e) => {
    selectedFiles = Array.from(e.target.files);
    if (selectedFiles.length >= 2) {
        // Validate all files
        for (const file of selectedFiles) {
            const val = validateFile(file);
            if (!val.valid) {
                showError(`File "${file.name}": ${val.message}`);
                selectedFiles = [];
                mosaicBtn.disabled = true;
                return;
            }
        }
        mosaicBtn.disabled = false;
        mosaicResults.classList.remove('hidden');
        mosaicImg.src = '';
        matchVisImg.src = '';
    } else {
        mosaicBtn.disabled = true;
        if (selectedFiles.length > 0) {
            showError('Please select at least 2 overlapping images for mosaicking.');
        }
    }
});

mosaicBtn.addEventListener('click', async () => {
    if (selectedFiles.length < 2) return;

    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });

    mosaicBtn.disabled = true;
    mosaicLoader.classList.remove('hidden');
    updateLoader(mosaicLoader, 'Detecting A-KAZE features...');

    try {
        updateLoader(mosaicLoader, 'Stitching frames into mosaic...');
        const response = await fetch('/mosaic', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Mosaicking failed');
        }

        const data = await response.json();
        mosaicImg.src = data.mosaic;
        matchVisImg.src = data.matching;
        mosaicDownloadBtn.href = data.mosaic;
        if (mosaicProcessingTime) {
            mosaicProcessingTime.textContent = `Mosaicking completed in ${data.processing_time}s`;
        }
        mosaicLoader.classList.add('hidden');
    } catch (error) {
        console.error(error);
        showError(error.message);
        mosaicLoader.classList.add('hidden');
        mosaicBtn.disabled = false;
    }
});
