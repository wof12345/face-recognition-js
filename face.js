const video = document.getElementById("video");
const container = document.querySelector(`.mainCont`);
const message = document.querySelector(`.message`);

let lastVideoIntervalInstance;

let detectedDescriptorSamples = [];
let predefindeDescriptorSamples = [];

let predefinedImageArray = ["./image/me.jpg", "./image/t.jpg"];

let detectorTracker = {
  unknown: 0,
  iteration: 0,
};
let faceDetected = false;

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
  faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
]).then(startVideo);

function startVideo() {
  predefinedImageArray.forEach((elm, ind) => {
    getImageData(elm, ind);
  });
  message.textContent = "Loading...";
  navigator.getUserMedia(
    { video: {} },
    (stream) => (video.srcObject = stream),
    (err) => console.error(err)
  );
}

video.addEventListener("play", async () => {
  const canvas = faceapi.createCanvasFromMedia(video);
  container.append(canvas);
  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  lastVideoIntervalInstance = setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors();

    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    faceapi.draw.drawDetections(canvas, resizedDetections);

    if (detectorTracker.iteration < 20) {
      if (!faceDetected) {
        message.textContent =
          "Detecting face... Please position your face at the center of the camera.";
        if (detections[0]) {
          detectedDescriptorSamples.push(detections[0].descriptor);

          detectorTracker.iteration++;
        } else
          message.textContent =
            "face not found... Please position your face at the center of the camera.";
      }
    } else testSamples();
  }, 60);
});

function testSamples() {
  predefindeDescriptorSamples.forEach((element, index) => {
    detectedDescriptorSamples.forEach((elm, ind) => {
      findBestMatch(elm, element, `person ${index}`);
    });
    console.log(detectorTracker[`person ${index}`]);

    if (detectorTracker[`person ${index}`] > detectorTracker["unknown"]) {
      faceDetected = true;
      message.textContent = `face match for person ${index}! redirecting...`;
      console.log(detectorTracker);

      console.log(predefindeDescriptorSamples);

      setTimeout(() => {
        // window.location = "http://localhost:8000/login/admin/123456";
        // container.style.display = "none";
      }, 4000);
      clearInterval(lastVideoIntervalInstance);
      lastVideoIntervalInstance = null;
    } else {
      if (!faceDetected) message.textContent = "face mismatch!";
    }
  });
  resetTracker();
}

function findBestMatch(data1, data2, person) {
  if (data1 !== undefined && data2 !== undefined) {
    const threshold = 0.6;
    const faceMatcher = new faceapi.FaceMatcher(data1, threshold);

    let data = faceMatcher.findBestMatch(data2);

    if (data._label === "person 1") detectorTracker[person]++;
  }
}

function resetTracker() {
  detectorTracker.iteration = 0;
  predefindeDescriptorSamples.forEach((element, index) => {
    detectorTracker[`person ${index}`] = 0;
  });
  detectorTracker.unknown = 0;
  detectedDescriptorSamples = [];
}

async function getImageData(imagePath, index) {
  let img = await faceapi.fetchImage(imagePath); //face image to train
  const imageData = await faceapi
    .detectSingleFace(img)
    .withFaceLandmarks()
    .withFaceDescriptor();

  // console.log(imageData);
  let faceDescription = faceapi.resizeResults(imageData, img).descriptor;

  predefindeDescriptorSamples.push(faceDescription);
  detectorTracker[`person ${index}`] = 0;

  console.log(predefindeDescriptorSamples);
  console.log(detectorTracker);
}

//
// let imageStore = [];
// function preview_image(event) {
//   document.getElementById("output_image").style.display = "block";
//   var reader = new FileReader();
//   reader.onload = function () {
//     var output = document.getElementById("output_image");

//     imageStore.push(reader.result);
//     console.log(reader, imageStore);
//     output.src = imageStore[0];
//   };
//   reader.readAsDataURL(event.target.files[0]);
// }
