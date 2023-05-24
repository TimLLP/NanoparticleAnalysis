function bubblecircle = randomSimulator(numBubbles,minRadius,maxRadius)
%%
global plotshow
% generate the properties of partical cluster
m = 890;
n = 1280;
nCircles = numBubbles;
circles = zeros(nCircles,3);
figure,hold on
axis image
set(gca,'XTick',[0:100:1300],'YTick',[0:100:900]);
title(['particles placed randomly'])

tic
for i=1:nCircles
    x = randi([1,n],1);
    y = randi([1,m],1);
    r = rand(1)*(maxRadius-minRadius)+minRadius;
    circles(i,[1,2,3]) = [y x r];
    circle(x,y,r)
end
rectangle('position',[0 0 n m],'edgecolor','r');
hold off;
toc

bubblecircle.xpos = circles(:,1);
bubblecircle.ypos = circles(:,2);
bubblecircle.radius= circles(:,3);

if plotshow==0
    close all
end


end
%%
function circle(x,y,r)
%x and y are the coordinates of the center of the circle
%r is the radius of the circle
%0.01 is the angle step, bigger values will draw the circle faster but
%you might notice imperfections (not very smooth)
ang=0:0.01:2*pi;
xp=r*cos(ang);
yp=r*sin(ang);
plot(x+xp,y+yp);
end