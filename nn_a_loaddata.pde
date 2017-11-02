Datum [] training_set;                                                                          // this uses the MNIST database of handwritten digits   
Datum [] testing_set;                                                                           // http://yann.lecun.com/exdb/mnist/ (accessed 04.06.09) 
Datum draw_set;
byte [] images;
byte [] labels;

//Images
//int inImgb =   16;
//int inLblb =    8;
//int tLabels =  7;
//int nPixelsX = 25;
//int nPixelsY = 20;
//int nImgsX =    5;
//int nImgsY =    4;
//int nCounts =   50;

//Numbers
int inImgb =   16;
int inLblb =    8;
int tLabels =  10;
int nPixelsX = 14;
int nPixelsY = 14;
int nImgsX =    10;
int nImgsY =    10;
int nCounts =   100;

//Both
int sPixelsX = 25;
int sPixelsY = 25;
int n;
int tPixels =nPixelsX*nPixelsY;
int sImgsX = sPixelsX*nPixelsX;
int sImgsY = sPixelsY*nPixelsY;
int nImgsperCount = nImgsX*nImgsY;
int nTotImgs =      nImgsX*nImgsY*nCounts;
float strokeLin = 15;
float strokePixel = 1;

void loadfi() {
  images = loadBytes("t10k-images-14x14.idx3-ubyte");
  labels = loadBytes("t10k-labels.idx1-ubyte");
  //images = loadBytes("n25x20.idx3-ubyte");
  //labels = loadBytes("l2.idx3-ubyte");
}
void loadtraind(int origin, int number) {
  training_set = new Datum [number];
  for (int i = 0; i < number; i++) {
    training_set[i] = new Datum(inImgb, tPixels, inLblb, tLabels);
    training_set[i].imageLoad(images, i+origin);
    training_set[i].labelLoad(labels, i+origin);
  }
}
void loadtestd(int origin, int number) {
  testing_set = new Datum [number];
  for (int i = 0; i < number; i++) {
    testing_set[i] = new Datum(inImgb, tPixels, inLblb, tLabels);
    testing_set[i].imageLoad(images, i+origin);                              
    testing_set[i].labelLoad(labels, i+origin);
  }
}
void loaddraw() {
  draw_set = new Datum(inImgb, tPixels, inLblb, tLabels);
  draw_set.imageLoad(bytesOutDrawing, 0);
  printArray(bytesOutDrawing);
}
class Datum {                                                                                   //DATUM: one number                                              
  float [] inputs;
  float [] outputs;
  int output;
  int inbytes;
  int bytesperimage;
  int inbyteslab;
  int bytesperlabel;
  Datum(int _inbytes, int _bytesperimage, int _inbyteslab, int _bytesperlabel) {
    inbytes = _inbytes;
    bytesperimage = _bytesperimage;
    bytesperlabel = _bytesperlabel;
    inbyteslab=_inbyteslab;
    inputs = new float [bytesperimage];                                                        //Input is 196 pixels each with a saturation number
    outputs = new float[bytesperlabel];                                                        //Output is Label number all numbers that are not the number ar -1 others are 1
  }
  void imageLoad(byte [] images, int num ) {                                                   // note, you must use int() to convert rather than (int) to cast:
    for (int i = 0; i < tPixels; i++){
      inputs[i] = int(images[i+(inbytes+num*bytesperimage)]) / 128.0 - 1.0;                    //There are  16 initial bytes in the images
    }
  }
  void labelLoad(byte [] labels, int num) {
    output = int(labels[inbyteslab+num]);                                                       // There is 8 initial bytes in the label 
    for (int i = 0; i < bytesperlabel; i++) {
      if (i == output) outputs[i] = 1.0; 
      else outputs[i] = -1.0;
    }
  }
}