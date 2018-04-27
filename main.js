/* create a new "console" entry in the virutal console
 * @param {String} data - data to print to virutal console
 * @param {Boolean} error - whether to log as an error or normal, default false
 */
function virtualConsoleLog(data, error){
  const date = new Date().toLocaleString();
  let pageConsole = document.getElementById("model-progress");
  let newData = document.createElement('span');
  newData.innerHTML = "[" + date + "] " + data;
  newData.className = "log-text" + (error ? " log-error":"");
  pageConsole.appendChild(newData);
  pageConsole.scrollTop = pageConsole.scrollHeight;
}

const LSTM_LAYERS = 2;
const LSTM_SIZE = 128;

let model;
let globalVocab;

document.addEventListener('DOMContentLoaded', () => {

  document.getElementById("train-model").addEventListener('mousedown', async () => {
    let inputText = document.getElementById("input-text").value;
    let seqLength = parseInt(document.getElementById("seq-length").value);
    let outputKeepProb = parseInt(document.getElementById("output-keep-prob").value);
    let epochs = parseInt(document.getElementById("epochs").value);
    let batchSize = parseInt(document.getElementById("batch-size").value);

    if(!inputText) {
      virtualConsoleLog("no data in input... stopping", true);
    }
    else {
      // set up training data
      let [trainIn, trainOut, vocab, indexToVocab] = prepareData(inputText, seqLength);
      globalVocab = vocab;
      trainIn = tf.tensor(trainIn);
      trainOut = tf.tensor(trainOut);

      // set up model
      model = new LSTM({
        seqLength: seqLength,
        outputKeepProb: outputKeepProb,
        vocab: vocab,
        indexToVocab: indexToVocab,
        numLayers: LSTM_LAYERS,
        hiddenSize: LSTM_SIZE
      });

      // create model
      await model.init();

      // train model
      await model.train(trainIn, trainOut, {
        batchSize: batchSize,
        epochs: epochs
      });
    }
  });

  document.getElementById("generate-text").addEventListener('mousedown', async() => {
    let primer = document.getElementById("primer").value;
    let predictLength = parseInt(document.getElementById("num-chars").value);

    if(model){
      let predicted = await model.predict(oneHotString(primer, globalVocab), predictLength);
      document.getElementById("output-text").value = predicted;
    }
    else {
      throw new Error("Model is not trained.")
    }
  });

  document.getElementById("keep-training").addEventListener('mousedown', async() => {
    // TODO implement later
    // let epochs = parseInt(document.getElementById("epochs").value);
    // let batchSize = parseInt(document.getElementById("batch-size").value);

    // if(model){
    //   await model.train(trainIn, trainOut, {
    //     batchSize: batchSize,
    //     epochs: epochs
    //   });
    // }
    // else {
    //   throw new Error("Model is not initially trained.")
    // }
  });
});