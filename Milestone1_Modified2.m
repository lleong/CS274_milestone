%y = cosu	0<= u <=2*pi

w1 = 1
b1 = 0.01
w2 = [1, 2, 3]
b2 = [0.01, 0.02, 0.03]
w3 = [3, 6, 9]
b3 = 0.03
z1 = [];
z = [];
output = [];
input = [];
desiredOp = [];

learning_rate = 0.1

function f = g(s)
	f= (1-exp(-2*s)) / (1+ exp(-2*s));
end

function f1 = g1(s)
	f1 = (4 * exp(-2*s)) / ( 1+ exp(-2*s))^2;
end

% Random Number Gen for Group1
%it generates random value between 0 and 2pi
function wq = randomGen()
        N = 1;
        lower = 0;
        upper = 8;
        interval = lower:upper;
        permuted = interval (randperm (length (interval)));
        wq = permuted (1:N)*rand(1);
        if wq > 6.28
            wq = wq* rand(1);
        endif
end



count = 0;

u = 0;
dataCount = 0;

while(dataCount < 21)

	u = dataCount * ((2 * pi) / 20)
	target = cos(u)
	error = 1;
	
	while ( error > 0.05 || error < - 0.05 )

		% input layer
		v1 = w1 * u + b1;
		v = g(v1);

		%hidden layer
		k = 1;
		while (k <= 3)
			z1(k) = w2(k) * v + b2(k);
			z(k) = g(z1(k));
			k++;
		end

		%output layer
		x = 1;
		y1 = 0;
		while ( x <= 3)
			y1 += w3(x) * z(x) + b3;
			x++;
		end

		y = g(y1);

		%% update weights value in output/ input and hidden layer
		b1_t = 0;	
		w1_t = 0;
		t = 1;
		
		error = y - target;
		while (t <= 3)
		  
		  b3 = b3 - (learning_rate * error * g1(y1));
		  
		  %changed y to y1
		  w3(t) = w3(t) - learning_rate * error * g1(y1)* z(t);

			%weight in the hidden layer:
			%Changed y to y1
		 b2(t) = b2(t) - (learning_rate * error * g1(y) * w3(t) * g1(z1(t)));
			
			%Changed y to y1
			c = g1(y1) * w3(t)* g1(z1(t));
			w2(t) = w2(t) - learning_rate * error * c * v;

			%weight in the input layer:
		  b1_t += c * w2(t) * g1(v1);
			w1_t += c * w2(t) * g1(v1) * u;
			t++;
		end

		w1 = w1 - learning_rate * error * w1_t;
		b1 = b1 - learning_rate * error * b1_t;
		count++;
	end
	
	Final_error = error
	Final_output = y
	No_Of_Iterations = count
	
	dataCount++;
	
	output(dataCount) = y;
	input(dataCount) = u;
	desiredOp(dataCount) = target;
	
	count = 0;
	
end

plot(input, desiredOp, input, output);