
w1 = rand;
b1 = rand;
w2 = rand(3,1);%[1 2 2]';
b2 = rand(3,1);%[0.01  0.02 0.04]';
w3 = rand(3,1);%[3 1 2]';
b3 = rand;

%numIters=100;


function y=NeuNet(u,w1,b1,w2,b2,w3,b3)
v1 = w1 * u + b1;
v=(1-exp(-2*v1)) / (1+ exp(-2*v1));
z1=w2*v+b2;
z=(1-exp(-2*z1)) ./ (1+ exp(-2*z1));
y1=w3'*z+b3;
y=(1-exp(-2*y1)) / (1+ exp(-2*y1));
end

function [w1,b1,w2,b2,w3,b3]=update_BPWeights(u,w1,b1,w2,b2,w3,b3)
yd=cos(u);
learning_rate = 0.1;
v1 = w1 * u + b1;
v=(1-exp(-2*v1)) / (1+ exp(-2*v1));
z1=w2*v+b2;
z=(1-exp(-2*z1)) ./ (1+ exp(-2*z1));
y1=w3'*z+b3;
y=(1-exp(-2*y1)) / (1+ exp(-2*y1));
error=y-yd;
G1_y1=(4*exp(-2*y1))/ ((1+ exp(-2*y1))^2);
G1_z=(4*exp(-2*z)) ./ ((1+ exp(-2*z)).^2);
G1_v=(4*exp(-2*v)) ./ ((1+ exp(-2*v)).^2);

b3=b3-learning_rate*error*G1_y1;
w3=w3-learning_rate*error*G1_y1*z;

b2=b2-learning_rate*error*G1_y1*(w3.*G1_z);
w2=w2-learning_rate*error*G1_y1*(w3.*G1_z)*v;

b1=b1-learning_rate*error*G1_v*G1_y1*(w3.*G1_z)'*w2;
w1=w1-learning_rate*error*G1_v*G1_y1*(w3.*G1_z)'*w2*u;
end



DATA=2*pi*[0:20]/20;
TARGET=cos(DATA);

TrainingSamples=25;
for m=1:TrainingSamples
InputIndex=floor(rand*21)+1;
%InputIndex=m;
u=DATA(InputIndex);
yd=TARGET(InputIndex);
y=NeuNet(u,w1,b1,w2,b2,w3,b3);
error=y-yd;
iters(m)=1;
while (abs(error)>0.05)% && iters<numIters)
[w1,b1,w2,b2,w3,b3]=update_BPWeights(u,w1,b1,w2,b2,w3,b3);
y=NeuNet(u,w1,b1,w2,b2,w3,b3);
error=y-yd;
iters(m)+=1;
end
sampleError(m)=error;
end
sampleError
iters

SqError=0;
for k=1:21
  y=NeuNet(DATA(k),w1,b1,w2,b2,w3,b3);
  SqError+=(y-TARGET(k))^2;
end
D=sqrt(SqError)/21;


SqError=0;
NN=1001;
for k=1:NN
  u=2*pi*(k-1)/NN;
  y=NeuNet(u,w1,b1,w2,b2,w3,b3);
  SqError+=(y-cos(u))^2;
end
D=sqrt(SqError)/NN;
