

void setup() {
  size(1300, 450);
  loadfi();
  loadtraind(0, nTotImgs);
  loadtestd(0, nTotImgs);
  setupdr();
  setupnn();
  trainnn(nTotImgs);
}
void draw() {
  background(255);
  drawdr();
  textSize(20);
  fill(100);
  text("NEURAL NETWORK IMPLEMENTATION", 240,30);
  drawnn();
  saveFrame("movie/####.png");
}
void keyPressed() {
  if (key == 't') testnn();
  if (key == 'd') {
    loaddraw();
    testnnd();
  }
  if (key == 'e') {
    setupdr();
  }
}