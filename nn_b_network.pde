Network nn;  

PVector nnl1origin = new PVector (500,50);
PVector nnl2origin = new PVector (900,50);
PVector nnl3origin = new PVector (900,200);//NEURAL NETWORK
int response, actual;                                                                                           //Response Value

void setupnn() {
  nn = new Network(tPixels, tPixels/4, tLabels, nPixelsX, nPixelsY, sPixelsX, sPixelsY);
}
void trainnn(float trains) {
  for (int i = 0; i < trains; i++) {                                                                            // this allows some fast training without displaying:
    int randomnet = int(random(0, training_set.length));                                                        // select a random training input and train
    response = nn.netrespond(training_set[randomnet].inputs);
    actual = training_set[randomnet].output;
    nn.train(training_set[randomnet].outputs);
  }
}
void testnn() {
  int row = int(random(0, testing_set.length));
  response = nn.netrespond(testing_set[row].inputs);
  actual = testing_set[row].output;
}
void testnnd() {
  response  =  nn.netrespond (draw_set.inputs);
}
void drawnn() {
  nn.drawl1(nnl1origin);
  nn.drawl2(nnl2origin);
  nn.drawl3(nnl3origin);
  fill(0);
  textAlign(CENTER);
  textSize(18);
  fill(100);
  text("neural network response:"+str(response), nnl3origin.x+120, nnl3origin.y + sPixelsY/2 +40);
   textSize(100);
  text(str(response), nnl3origin.x+120, nnl3origin.y + sPixelsY/2 +150);
}
class Network {                                                                                                // This class is for the neural network, which is hard coded with three layers: input, hidden and output
  Neuron [] m_input_layer;                                                                                     // this network is hard coded to only have one hidden layer
  Neuron [] m_hidden_layer;
  Neuron [] m_output_layer;                                                                                    // create a network specifying numbers of inputs, hidden layer neurons
  int nPixX, nPixY;
  float sPixX, sPixY;

  Network(int inputs, int hidden, int outputs, int _nPixX, int _nPixY, float _sPixX, float _sPixY) {                                                               // and number of outputs, e.g. Network(4,4,3)
    m_input_layer = new Neuron [inputs];
    m_hidden_layer = new Neuron [hidden];
    m_output_layer = new Neuron [outputs];
    for (int i = 0; i < m_input_layer.length; i++) m_input_layer[i] = new Neuron();                            // set up the network topology   
    for (int j = 0; j < m_hidden_layer.length; j++) m_hidden_layer[j] = new Neuron(m_input_layer);             // route the input layer to the hidden layer  
    for (int k = 0; k < m_output_layer.length; k++) m_output_layer[k] = new Neuron(m_hidden_layer);            // route the hidden layer to the output layer
    nPixX = _nPixX;
    nPixY = _nPixY;
    sPixX = _sPixX;
    sPixY = _sPixY;
  }
  int netrespond(float [] inputs) {
    float [] responses = new float [m_output_layer.length];                                                    // feed forward    // simply set the input layer to display the inputs
    for (int i = 0; i < m_input_layer.length; i++)  m_input_layer[i].m_output = inputs[i];                     // now feed forward through the hidden layer
    for (int j = 0; j < m_hidden_layer.length; j++) m_hidden_layer[j].neuro_respond();
    for (int k = 0; k < m_output_layer.length; k++) responses[k] = m_output_layer[k].neuro_respond();          // and finally feed forward to the output layer
    int response = -1;                                                                                         // now check the best response:
    float best = max(responses);
    for (int a = 0; a < responses.length; a++)  if (responses[a] == best) response = a;
    return response;
  }
  void train(float [] outputs) {
    for (int k = 0; k < m_output_layer.length; k++) {                                                           // adjust the output layer
      m_output_layer[k].finderror(outputs[k]);
      m_output_layer[k].train();
    }
    for (int j = 0; j < m_hidden_layer.length; j++) {                                                           // propagate back to the hidden layer
      m_hidden_layer[j].train();                                                                                // the input layer doesn't learn:   // it is simply the inputs
    }
  }
  void drawl1(PVector origin) {                                                                                 // note, this draw is hard-coded for Network(196,49,10)  // which reflects my use of the MNIST database of handwritten digits
    for (int i = 0; i < m_input_layer.length; i++) {

      pushMatrix();
      translate(origin.x, origin.y);
      translate((i%nPixX) *sPixX, (i/nPixX) * sPixY);
      float level = (0.5-(m_input_layer[i].m_output*0.5)); 
      stroke(200);
      fill(255 * level);
      rect(0, 0, sPixX, sPixY);
      popMatrix();
      stroke(100);
      noFill();
      rect(origin.x, origin.y, sPixX*nPixX, sPixY*nPixY);
    }
  }
  void drawl2(PVector origin) {
    for (int j = 0; j < m_hidden_layer.length; j++) {
      pushMatrix();
      translate(origin.x, origin.y);
      translate((j%nPixX) * sPixX, (j/nPixX) * sPixY);
      float level = (0.5-(m_hidden_layer[j].m_output*0.5)); 
      fill(255 * level);
      rect(0, 0, sPixX, sPixY);
      popMatrix();
    }
  }
  void drawl3(PVector origin) {
    for (int k = 0; k < m_output_layer.length; k++) {                                                           // this is slightly tricky -- I've switched the order so the output   
      pushMatrix();                                                                                             // neurons are arrange 1,2,3...8,9,0 rather than 0,1,2...7,8,9
      translate(origin.x, origin.y);
      translate(k*sPixX, 0);                                                                                    // (that's what the (k+9) % 10 is doing)
      float level = (0.5-(m_output_layer[k].m_output*0.5)); 
      fill(255 * level);
      rect(0, 0, sPixX, sPixY);                                                                                 //OUTPUT WINNER
      popMatrix();
    }
  }
}
class Neuron {                                                                                                  // This class is for each neuron.  It works // as a feed-forward multilayer perceptron, // learning by backpropagation
  Neuron [] m_inputs;
  float [] m_weights;
  float m_threshold;
  float m_output;
  float m_error;

  Neuron() {                                                                                                   // the input layer of neurons have no inputs:
    m_threshold = 0.0;
    m_error = 0.0;
    m_output = sigm(random(-5.0, 5.0));                                                                        // initial random output
  }
  Neuron(Neuron [] inputs) {                                                                                   // all other layers (hidden and output) have  // neural inputs
    m_inputs = inputs;  
    m_weights = new float [inputs.length];
    for (int i = 0; i < inputs.length; i++) m_weights[i] = random(-1.0, 1.0);
    m_threshold = random(-1.0, 1.0);
    m_error = 0.0;
    m_output = sigm(random(-5.0, 5.0));                                                                        // initial random output
  }
  float neuro_respond() {                                                                                      // respond looks at the layer below, and prepares a response:
    float input = 0.0;
    for (int i = 0; i < m_inputs.length; i++) input += m_inputs[i].m_output * m_weights[i];
    m_output = sigm(input + m_threshold);                                                                      // reset our error value ready for training  
    m_error = 0.0;
    return m_output;
  }
  void finderror(float desired) {                                                                              // find error is used on the output neurons
    m_error = desired - m_output;
  }
  void train() {                                                                                               // train adjusts the weights by comparing actual output to correct output
    float LEARNING_RATE = 0.01;    
    float delta = (1.0 - m_output) * (1.0 + m_output) * m_error * LEARNING_RATE;
    for (int i = 0; i < m_inputs.length; i++) {        
      m_inputs[i].m_error += m_weights[i] * m_error;                                                          // tell the next layer down what it's doing wrong
      m_weights[i] += m_inputs[i].m_output * delta;                                                           // correct our weights
    }
  }
}
float sigm(float x) {                                                                                         // a sigmoid function is the neuron's response to inputs, the sigmoidal response ranges from -1.0 to 1.0 for example,
  return  2.0 / (1.0 + exp(-2.0 * (x/5))) - 1.0;                                                              // the weighted sum of inputs might be "2.1" our response would be lookupSigmoid(2.1) = 0.970 this is a look up table
}                                                                                                             //  for sigmoid (neural response) values which is valid from -5.0 to 5.0