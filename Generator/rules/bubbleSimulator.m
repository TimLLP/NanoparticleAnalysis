function  bubblecircle = bubbleSimulator(numBubbles,minRadius,maxRadius,gap)
% close all
% parameters
% numBubbles=500;     % number of bubbles
% minRadius=35;      % minimum radius
% maxRadius=45;     % maximum radius
% gap = 5;
global plotshow
%%
numBubbles=numBubbles;
% other parameters
k=100;               % spring constant
c=1;                % damping constant
cg=0.1;             % damping w.r.t. ground
g=5;                % gravity
Ts=0.005;           % simulation sampling time
Ttal=10;             % simulation total time
% fixed parameter
m=890/2;
n=1280/2;

% generate bubbles and initialise
bubbles.radius=rand(numBubbles,1)*(maxRadius-minRadius)+minRadius;
%bubbles.radius = radius;
bubbles.mass=0.1*ones(numBubbles,1);
bubbles.pos=(rand(numBubbles,2)-0.5)*sqrt(2)*sqrt(numBubbles)*maxRadius;
bubbles.vel=zeros(numBubbles,2);

%%
% offline computation
rr=repmat(bubbles.radius,1,numBubbles);
bubbles.sumRadius=(rr+rr');
tic
t=0;
circle(bubbles.pos(:,1),bubbles.pos(:,2),bubbles.radius);
axis equal;drawnow;
title(['particles placed iteratively'])
% simulation starts
while t<Ttal
    t=t+Ts;
    bubbles=updatePosition(bubbles,numBubbles,Ts,k,c,cg,g,gap);
    if ~mod(round(t/Ts),10)
        circle(bubbles.pos(:,1),bubbles.pos(:,2),bubbles.radius);
        axis equal;drawnow;
    end
end
hold on
rectangle('position',[-n -m 2*n 2*m],'edgecolor','r');
hold off


blx = find(bubbles.pos(:,1)<n+minRadius/2  & bubbles.pos(:,1)>-n-minRadius/2);
bly = find(bubbles.pos(blx,2)<m+minRadius/2 & bubbles.pos(blx,2)>-m-minRadius/2);
bl = blx(bly);

bubblecircle.xpos = bubbles.pos(bl,2)+890/2;
bubblecircle.ypos = bubbles.pos(bl,1)+1280/2;
bubblecircle.radius= bubbles.radius(bl);

toc
%%
figure;
%draw a figure
xlim([0,1280]);
ylim([0,890]);
axis image
set(gca,'XTick',[0:100:1300],'YTick',[0:100:900]);
hold on
title(['particles placed all with spring-damper-mass'])
for ii = 1:length(bl)
    circlestep(bubblecircle.ypos(ii),bubblecircle.xpos(ii),bubblecircle.radius(ii));
end
rectangle('position',[0 0 2*n 2*m],'edgecolor','r');
hold off

if plotshow==0
    close all
end


end
%%


function bubbles=updatePosition(bubbles,numBubbles,Ts,k,c,cg,g,gap)
% consider elasticity, damping relative to ground and attraction to the centre
posx=bubbles.pos(:,1);
posy=bubbles.pos(:,2);
relPosx=repmat(posx',numBubbles,1)-repmat(posx,1,numBubbles);
relPosy=repmat(posy',numBubbles,1)-repmat(posy,1,numBubbles);
relDist=sqrt(relPosx.^2+relPosy.^2);
overlapDist=bubbles.sumRadius-relDist;
overlapDist(1:numBubbles+1:numBubbles*numBubbles)=-inf;
velx=bubbles.vel(:,1);
vely=bubbles.vel(:,2);
relVelx=repmat(velx',numBubbles,1)-repmat(velx,1,numBubbles);
relVely=repmat(vely',numBubbles,1)-repmat(vely,1,numBubbles);

% calculate forces and acceleration
N=overlapDist>gap;
F=zeros(numBubbles,2);
C=zeros(numBubbles,2);
ids=find(sum(N,2)); % find balls that have at least one neighbour
% relative spring force
normedF=N(ids,:).*overlapDist(ids,:)./relDist(ids,:);
F(ids,1)=sum(normedF.*-relPosx(ids,:),2,'omitnan')*k;
F(ids,2)=sum(normedF.*-relPosy(ids,:),2,'omitnan')*k;
% relative damping force
C(ids,1)=sum(N(ids,:).*relVelx(ids,:),2,'omitnan')*c;
C(ids,2)=sum(N(ids,:).*relVely(ids,:),2,'omitnan')*c;
% ground damping and gravity force
Cg=-cg*bubbles.vel;
G=-g*bubbles.pos*0.05;
accel=(F+C+Cg)./repmat(bubbles.mass,1,2)+G;

% update states
bubbles.pos=Ts*bubbles.vel+bubbles.pos;
bubbles.vel=Ts*accel+bubbles.vel;
end
%%
function circle(x,y,r,numPoint,plotOpt)
% CIRCLE(x,y,r) plots a circle of radius r around point (x,y).
%
% x and y are the coordinates of the center of the circle
% r is the radius of the circle
% 0.01 is the angle step, bigger values will draw the circle faster but
% you might notice imperfections (not very smooth)

if nargin==3
    plotOpt='.';
    numPoint=51;
elseif nargin==4
    plotOpt='-.';
end

x=x(:);
y=y(:);
r=r(:);
numCircle=length(x);
theta=linspace(0,2*pi,numPoint)';
xp=reshape(repmat(r,1,numPoint),numCircle*numPoint,1).*repmat(cos(theta),numCircle,1);
yp=reshape(repmat(r,1,numPoint),numCircle*numPoint,1).*repmat(sin(theta),numCircle,1);
plot(repmat(x,numPoint,1)+xp,repmat(y,numPoint,1)+yp,plotOpt);
end


function circlestep(x,y,r)
%x and y are the coordinates of the center of the circle
%r is the radius of the circle
%0.01 is the angle step, bigger values will draw the circle faster but
%you might notice imperfections (not very smooth)
ang=0:0.01:2*pi;
xp=r*cos(ang);
yp=r*sin(ang);
plot(x+xp,y+yp);
end


