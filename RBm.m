clc
N=3;
M=4;%1,2,4,8
eta=0.001;
Pattern=[-1 -1 -1
    1 -1 1
    -1 1 1 
    1 1 -1];
W=normrnd(0,0.577,M,N);
a=0;
b=0;
c=0;
d=0;
v=zeros(1,N);
h=zeros(1,M);
THETAv=zeros(1,N);
THETAh=zeros(1,M);

    for trial=1:100000
    mu= randi([1 4]);
    v=Pattern(mu,:);
    v0=v;
    bh=v*W'-THETAh;
    bh0=bh;
       for i=1:M
           r=rand;
           if r<(1+exp(-bh(1,i)))^-1
               h(1,i)=1;
           else
               h(1,i)=-1;
           end
       end
      
       for t=1:200    
       
          bv=h*W-THETAv;
        for j=1:N
           r=rand;
           if r<(1+exp(-bv(1,j)))^-1
               v(1,j)=1;
           else
               v(1,j)=-1;
           end
       end
              bh=v*W'-THETAh;
            for i=1:M
                 r=rand;
                if r<(1+exp(-bh(1,i)))^-1
               h(1,i)=1;
                else
               h(1,i)=-1;
                end
            end


       end


deltaW=eta*((tanh(bh0))'*v0-(tanh(bh))'*v);
deltaTHETAv=-eta*(v0-v);
deltaTHETAh=-eta*((tanh(bh0))-(tanh(bh)));

W=W+deltaW;
THETAv=THETAv+deltaTHETAv;
THETAh=THETAh+deltaTHETAh;

    end

v1=zeros(1,3);
e=rand;
for k=1:3
   if rand>0.5
       v1(1,k)=1;
   else
       v1(1,k)=-1;
   end
end
 for trial=1:10000
    
    v=v1;
    
    bh=v*W'-THETAh;
   
       for i=1:M
           r=rand;
           if r<(1+exp(-bh(1,i)))^-1
               h(1,i)=1;
           else
               h(1,i)=-1;
           end
       end
      
       for t=1:200    
       
          bv=h*W-THETAv;
        for j=1:N
           r=rand;
           if r<(1+exp(-bv(1,j)))^-1
               v(1,j)=1;
           else
               v(1,j)=-1;
           end
       end
              bh=v*W'-THETAh;
            for i=1:M
                 r=rand;
                if r<(1+exp(-bh(1,i)))^-1
               h(1,i)=1;
                else
               h(1,i)=-1;
                end
            end


       end
 if v==Pattern(1,:)
       a=a+1;
 end
 if v==Pattern(2,:)   
       b=b+1;
 end
 if v==Pattern(3,:)
       c=c+1;
 end
 if v==Pattern(4,:)
       d=d+1;
 end
 
 
 
 end

DKL=0.25*(log(0.25/(0.0001*a)))+0.25*(log(0.25/(0.0001*b)))+0.25*(log(0.25/(0.0001*c)))+0.25*(log(0.25/(0.0001*d)))




























