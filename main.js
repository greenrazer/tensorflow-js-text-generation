/* create a new "console" entry in the virutal console
 * @param {String} data - data to print to virutal console
 * @param {Boolean} error - whether to log as an error or normal, default false
 */
function virutalConsoleLog(data, error){
  const date = new Date().toLocaleString();
  let pageConsole = document.getElementById("model-progress");
  let newData = document.createElement('span');
  newData.innerHTML = "[" + date + "] " + data;
  newData.className = "log-text" + (error ? " log-error":"");
  pageConsole.appendChild(newData);
  pageConsole.scrollTop = pageConsole.scrollHeight;
}

document.addEventListener('DOMContentLoaded', () => {

  const trainButton = document.getElementById("train-model");
  trainButton.addEventListener('click', () => {
    let inputText = document.getElementById("input-text").value;
    if(!inputText) {
      virutalConsoleLog("no data in input... stopping", true);
    }
    else {
      let [inData, outData, vocab] = prepareData(inputText);
      let model = new LSTM({
        seqLength: 2,
        numLayers: 2,
        hiddenSize: 128,
        vocabSize: vocab.size,
        outputKeepProb: 1
      });

      model.init({
        logger:virutalConsoleLog
      }).then(() => {
        model.train(inData, outData);
      });
    }
  });

  const generateButton = document.getElementById("generate-text");
  generateButton.addEventListener('click', () => {
  });
})