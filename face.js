const video = document.getElementById("video");
const container = document.querySelector(`.mainCont`);
const message = document.querySelector(`.message`);

let lastVideoIntervalInstance;

let predefinedDescriptorSamples = [];
let detectedDescriptorSamples = [];

let detectorTracker = {
  "person 1": 0,
  unknown: 0,
  iteration: 0,
};

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
  faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
]).then(startVideo);

function startVideo() {
  message.textContent =
    "Detecting face... Please position your face at the center of the camera.";
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

  let img = await faceapi.fetchImage("./image/me.jpg"); //face image to train
  const imageData = await faceapi
    .detectSingleFace(img)
    .withFaceLandmarks()
    .withFaceDescriptor();

  let faceDescription = faceapi.resizeResults(imageData, img);
  lastVideoIntervalInstance = setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors();

    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    faceapi.draw.drawDetections(canvas, resizedDetections);

    console.log(detectorTracker);

    if (detectorTracker.iteration < 20) {
      if (detections[0]) {
        if (faceDescription) {
          predefinedDescriptorSamples.push(detections[0].descriptor);
          detectedDescriptorSamples.push(faceDescription.descriptor);

          detectorTracker.iteration++;
        }
      } else
        message.textContent =
          "face not found... Please position your face at the center of the camera.";
    } else testSamples();
  }, 60);
});

function testSamples() {
  predefinedDescriptorSamples.forEach((elm, ind) => {
    findBestMatch(elm, detectedDescriptorSamples[ind]);
  });

  if (detectorTracker["person 1"] > detectorTracker["unknown"]) {
    message.textContent = "face match! redirecting...";
    setTimeout(() => {
      window.location = "http://localhost:8000/login/admin/123456";
      container.style.display = "none";
    }, 4000);
    clearInterval(lastVideoIntervalInstance);
    lastVideoIntervalInstance = null;
  } else {
    message.textContent = "face mismatch!";
  }
  resetTracker();
}

function findBestMatch(data1, data2) {
  const threshold = 0.6;
  const faceMatcher = new faceapi.FaceMatcher(data1, threshold);

  let data = faceMatcher.findBestMatch(data2);

  detectorTracker[data._label]++;
}

function resetTracker() {
  detectorTracker.iteration = 0;
  detectorTracker["person 1"] = 0;
  detectorTracker.unknown = 0;
  predefinedDescriptorSamples = [];
  detectedDescriptorSamples = [];
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
