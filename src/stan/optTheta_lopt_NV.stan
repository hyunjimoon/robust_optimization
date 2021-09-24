data{
  int <lower = 0> n;
  int <lower = 0> p;
  matrix[n,p] X ;
  real y [n];
  real <lower = 0> profit;
  real <lower = 0> cost;
}
parameters{
  real theta;
}
model{
  target += profit * fmin(w, y) - cost * w
}