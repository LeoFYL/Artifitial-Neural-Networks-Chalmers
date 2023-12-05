 
clc;
p=12;%12;%24,48,70,100,120
 N=120;
 error=0;
 nTrails=100000;
for trail=1:nTrails
           P=zeros(p,N);
            for i=1:p 
            for j=1:N
                 s=rand;
                 if(s<0.5)
                 P(i,j)=1; 
                   else
                 P(i,j)=-1;
                 end
            end
            end
       
             w=zeros(N,N);
          for m=1:p
          pmu=P(m,:);
          w=w+pmu'*pmu/N;
          end
            
              for n=1:N
                  w(n,n)=0;
              end
                       
                     
%///Neuron update, using equation (2.28)
h= randi([1 p]);
y=P(h,:);
random_neuron = randi([1 120]);
b=zeros(1,120);
for q=1:120
    b(1,q)=y(1,random_neuron);
end
% (1-1/120)*y(1,random_neuron) +
  activation =  sign( (1-1/120)*y(1,random_neuron) +b*w(random_neuron, 1:120)');
     
  if activation == 0
  activation = 1;
  end
   

  if activation(1,1) ~= y(1,random_neuron)
 error = error + 1;
 end


 end
