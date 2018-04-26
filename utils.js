function prepareData(text, seqLength) {
  let data = text.split("");
  let vocab = getVocab(data);
  let dataX = [];
  let dataY = [];
  for (let i = 0; i < data.length - seqLength; i++){
    let inSeq = data.slice(i, i+seqLength);
    let outSeq = data[i+seqLength];
    dataX.push(inSeq.map(x=>oneHot(vocab.size, vocab.get(x))));
    dataY.push(oneHot(vocab.size, vocab.get(outSeq)));
  }
  return [dataX, dataY, vocab];
}

function oneHot(size, at){
  let vector = [];
  for(let i = 0; i < size; i++){
    if(at == i){
      vector.push(1);
    }
    else{
      vector.push(0);
    }
  }
  return vector;
}

function getVocab(arr) {
  //get letter mapped to amount of occurances
  let counts = new Map();
  for(let i of arr){
    if(counts.has(i)){
      const value = counts.get(i);
      counts.set(i, value+1);
    }
    else {
      counts.set(i, 1);
    }
  }
  // here we are taking those occurances and turning it in
  // into a map from letter to how frequetly it appears relative to other letters
  return new Map(Array.from(counts).sort((a, b) => {
    return b[1] - a[1];
  }).map((value, i) => {
    return [value[0], i];
  }));
}
