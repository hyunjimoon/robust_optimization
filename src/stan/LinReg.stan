data{
  int <lower = 0> n;
  int <lower = 0> p;
  matrix[n,p] X ;
  real y [n];
}
parameters{
  vector[p] beta;
}
model{
  y ~ normal(X * beta, 1);
}
