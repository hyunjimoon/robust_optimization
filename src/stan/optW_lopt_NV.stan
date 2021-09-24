data{
  int <lower = 0> n;
  vector[n] y;
  real <lower = 0> profit;
  real <lower = 0> cost;
}
parameters{
  real w;
}
model{
  for (i in 1:n){
    target += profit * fmin(w, y[i]) - cost * w;
  }
}