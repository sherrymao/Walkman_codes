function plot_network_ybc(Laplacian, Coordinates)

N_nodes = size(Coordinates,1);
%N_edges = size(Laplacian,1);

cm=0.95;

figure(1);
hold on;


for j=1:N_nodes
    for i =j+1:N_nodes
        if Laplacian(j,i)<0
            line([Coordinates(i,1);Coordinates(j,1)],[Coordinates(i,2);Coordinates(j,2)],'Color',[0.4 0.4 0.4],'LineWidth',2.2);
        end
    end
end

for i=1:N_nodes
    cm = norm([Coordinates(i,1),Coordinates(i,2)])./(sqrt(2));
    plot(Coordinates(i,1),Coordinates(i,2),'o','MarkerFaceColor',[1,Coordinates(i,1)./50,Coordinates(i,2)./50],'markersize',28,'MarkerEdgeColor', [0,0,0],'LineWidth',2);
    if i>9
        if i>99
             text(Coordinates(i,1)-0.8,Coordinates(i,2),num2str(i),'fontsize',16,'FontWeight','bold');
        else
            text(Coordinates(i,1)-0.5,Coordinates(i,2),num2str(i),'fontsize',16,'FontWeight','bold');%0.021
        end
    else
        text(Coordinates(i,1)-0.2,Coordinates(i,2),num2str(i),'fontsize',16,'FontWeight','bold');%0.013
    end
end
xmin = min(Coordinates(:,1));
xmax = max(Coordinates(:,1));
ymin = min(Coordinates(:,2));
ymax = max(Coordinates(:,2));

axis([xmin-0.05 xmax+0.05 ymin-0.05 ymax+0.05]);
box off; axis off;
set(gca,'XTick',[]);
set(gca,'YTick',[]);