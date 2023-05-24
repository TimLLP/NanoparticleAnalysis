function bubblecircle = CircleSimulator(numBubbles,minRadius,maxRadius,gap)
%%
global plotshow
% generate the properties of partical cluster
%fixed parameter
m = 890;
n = 1280;
gap
nCircles = numBubbles; % number of bubbles
% minRadius=35;      % minimum radius
% maxRadius=45;     % maximum radius
circles = zeros(nCircles,3);
% for better view
figure,hold on
axis image
set(gca,'XTick',[0:100:1300],'YTick',[0:100:900]);
title(['particles placement one by one'])
tic
for i=1:nCircles
    %Flag which holds true whenever a new circle was found
    newCircleFound = false;
    isblank = false;
    %loop iteration which runs until finding a circle which doesnt intersect with previous ones
    while ~newCircleFound
        i
        x = randi([1,n],1);
        y = randi([1,m],1);
        %r = 10+10*rand(1);
        r = rand(1)*(maxRadius-minRadius)+minRadius;
        
        %calculates distances from previous drawn circles
        prevCirclesY = circles(1:i-1,1);
        prevCirclesX = circles(1:i-1,2);
        prevCirclesR = circles(1:i-1,3);
        distFromPrevCircles = ((prevCirclesX-x).^2+(prevCirclesY-y).^2).^0.5;
        
        %if the distance is not to small - adds the new circle to the list
        if i==1 || sum(distFromPrevCircles<=(r+prevCirclesR-gap))==0
            newCircleFound = true;
            circles(i,[1,2,3]) = [y x r];
            circle(x,y,r)
        end
    end
end
toc
rectangle('position',[0 0 n m],'edgecolor','r');
hold off

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

