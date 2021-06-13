import logo from './logo.svg';
import './App.css';
import React from 'react';
import * as tf from '@tensorflow/tfjs';
import cat from './cat.jpg';
import {CLASSES} from './imagenet_classes';
const axios = require('axios');

const IMAGE_SIZE = 224;
let mobilenet;
let demoStatusElement;
let status;
let mobilenet2;

class App extends React.Component {

  constructor(props){
    super(props);
    this.state = {
      load:false,
      status: "F1 score of the model is: 0.9908 ",
      probab: ""
     };
     this.mobilenetDemo = this.mobilenetDemo.bind(this);
     this.predict = this.predict.bind(this);
     this.showResults = this.showResults.bind(this);
     this.filechangehandler = this.filechangehandler.bind(this);
  }

  async mobilenetDemo(){
    const catElement = document.getElementById('cat');
    if (catElement.complete && catElement.naturalHeight !== 0) {
      this.predict(catElement);
      catElement.style.display = '';
    } else {
      catElement.onload = () => {
        this.predict(catElement);
        catElement.style.display = '';
      }
    }

  };

  async predict(imgElement) {
    // The first start time includes the time it takes to extract the image
    // from the HTML and preprocess it, in additon to the predict() call.
    const startTime1 = performance.now();
    // The second start time excludes the extraction and preprocessing and
    // includes only the predict() call.
    let startTime2;
    let img = tf.browser.fromPixels(imgElement).toFloat().reshape([1, 224, 224, 3]);
    //img = tf.reverse(img, -1);
    this.setState({
            load:true
    });
    // const proxyurl = "https://cors-anywhere.herokuapp.com/";
    // const url = 'https://api.liveconnect.in/backend/web/erpsync/get-all-orders?data=dbCode=UAT04M%7Cidx=100%7CuserId=6214%7Cres_format=1'; // site that doesn’t send Access-Control-*
    // fetch(proxyurl + url).then((resp) => resp.json())
    //   .then(function(data) {
    //     console.log(data);
    //   })
    //   .catch(function(error) {
    //     console.log(error);
    //   }); 
    const image = await axios.post('http://localhost:5000/detect', {'image': img.dataSync()});
    // let url = "http://localhost:5000/detect";
    // // let url = "https://master-7rqtwti-z5pxqu5yvra7a.us-4.platformsh.site/detect";
    // const image = await fetch(url, {
    //   method: 'POST', // *GET, POST, PUT, DELETE, etc.
    //   mode: 'no-cors', // no-cors, *cors, same-origin
    //   body: JSON.stringify({'image': img.dataSync()}) // body data type must match "Content-Type" header
    // });
    this.setState({
            load:false
    });
    // // Show the classes in the DOM.
    this.showResults(imgElement, image.data['disease'], image.data['probab'], tf.tensor3d([image.data['image']].flat(), [224, 224, 3]));
  }


  async showResults(imgElement, diseaseClass, probab, tensor) {
    const predictionContainer = document.createElement('div');
    predictionContainer.className = 'pred-container';

    const imgContainer = document.createElement('div');
    imgContainer.appendChild(imgElement);
    predictionContainer.appendChild(imgContainer);

    const probsContainer = document.createElement('div');
    const predictedCanvas = document.createElement('canvas');
    probsContainer.appendChild(predictedCanvas);

    predictedCanvas.width = tensor.shape[0];
    predictedCanvas.height = tensor.shape[1];
    tensor = tf.reverse(tensor, -1);
    await tf.browser.toPixels(tensor, predictedCanvas);
    console.log(probab);
    this.setState({
      probab: "The last prediction was " + parseFloat(probab)*100 + " % accurate!"
    });
    const predictedDisease = document.createElement('p');
    predictedDisease.innerHTML = 'Disease: ';
    const i = document.createElement('i');
    i.innerHTML = CLASSES[diseaseClass];
    predictedDisease.appendChild(i);
    
    //probsContainer.appendChild(predictedCanvas);
    //probsContainer.appendChild(predictedDisease);

    predictionContainer.appendChild(probsContainer);
    predictionContainer.appendChild(predictedDisease);
    const predictionsElement = document.getElementById('predictions');
    predictionsElement.insertBefore(
        predictionContainer, predictionsElement.firstChild);
  }

  filechangehandler(evt){
    let files = evt.target.files;
    for (let i = 0, f; f = files[i]; i++) {
      // Only process image files (skip non image files)
      if (!f.type.match('image.*')) {
        continue;
      }
      let reader = new FileReader();
      reader.onload = e => {
        // Fill the image & call predict.
        let img = document.createElement('img');
        img.src = e.target.result;
        img.width = IMAGE_SIZE;
        img.height = IMAGE_SIZE;
        img.onload = () => this.predict(img);
      };

      // Read in the image file as a data URL.
      reader.readAsDataURL(f);
    }
  }

  componentDidMount(){
    this.mobilenetDemo();
  }

  render(){
    return (
        <div className="tfjs-example-container">
        <section className='title-area'>
          <h1>ResTS for Plant Disease Diagnosis</h1>
        </section>

        <section>
          <p className='section-head'>Description</p>
          <p>
            This WebApp uses the ResTS model which will be made available soon for public use.

            It is not trained to recognize images that DO NOT have BLACK BACKGROUNDS. For best performance, upload images of leaf\/Plant with black background. You can see the disease categories it has been trained to recognize in <a
              href="https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/segmented">this folder</a>.
          </p>
        </section>

        <section>
          <p className='section-head'>Status</p>
          {this.state.load?<div id="status">{this.state.status}</div>:<div id="status">{this.state.status}<br></br>{this.state.probab}</div>}
        </section>

        <section>
          <p className='section-head'>Model Output</p>

          <div id="file-container">
            Upload an image: <input type="file" id="files" name="files[]" onChange={this.filechangehandler} multiple />
          </div>
          {this.state.load?<div className="lds-roller"><div></div><div></div><div></div><div></div><div></div><div></div><div></div><div></div></div>:''}

          <div id="predictions"></div>

          <img id="cat" src={cat}/>
        </section>
      </div>


    );
  }
}

export default App;
