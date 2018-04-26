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

let model;
let seqLength = 2;
let vocab;
let indexToVocab;

document.addEventListener('DOMContentLoaded', () => {

  document.getElementById("train-model").addEventListener('click', () => {
    let inputText = document.getElementById("input-text").value;
    if(!inputText) {
      virtualConsoleLog("no data in input... stopping", true);
    }
    else {
      let inData;
      let outData;
      [inData, outData, vocab, indexToVocab] = prepareData(inputText, seqLength);
      inData = tf.tensor(inData);
      outData = tf.tensor(outData);
      if(!model){
        model = new LSTM({
          seqLength: seqLength,
          numLayers: 2,
          hiddenSize: 128,
          vocabSize: vocab.size,
          outputKeepProb: 1
        });

        model.init({
          logger:virtualConsoleLog
        }).then(() => {
          model.train(inData, outData, {
            logger:virtualConsoleLog,
            batchSize: 10,
            epochs: 10
          });
        });
      }
      else {

      }
    }
  });

  document.getElementById("generate-text").addEventListener('click', async() => {
    let primer = "ga"
    let predicted = await model.predict(oneHotString(primer, vocab), 100);
    let decoded = await decodeOutput(predicted, indexToVocab);
    document.getElementById("output-text").value = decoded;
  });
});