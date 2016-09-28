#include <bits/stdc++.h>
#include <vector>
using namespace std;

int rdtsc()
{
    __asm__ __volatile__("rdtsc");
    return 1;
}
template<class T>
void Bubblesort(vector<T>&vect){
    for(unsigned i=0;i<vect.size();i++){
        for(unsigned j=0;j<vect.size()-i-1;j++){
            T aux;
            if(vect[j]->distancia<vect[j+1]->distancia){
                aux=vect[j];
                vect[j]=vect[j+1];
                vect[j+1]=aux;

            }
        }
    }

}
template<class Tipo>
struct node
{
    Tipo data;
    list<node*> neighbors;
    bool verificado=0;
    float distancia;
};

template<class Tipo>
struct edge
{
    pair<node<Tipo>*,node<Tipo>*> xy;
    int weight=0;
};

template<class Tipo>
struct graph
{
    list<node<Tipo>> nodes;
    vector<edge<Tipo>> edges;
    list<node<Tipo>*> L;
    void add_node(Tipo x,float y)
    {
        node<Tipo> tmp;
        tmp.data=x;
        tmp.distancia=y;
        nodes.push_back(tmp);
    }

    void add_edge(Tipo x, Tipo y, int w)
    {
        edge<Tipo> tmp2;
        tmp2.weight=w;
        for(auto i=nodes.begin();i!=nodes.end();++i)
            if(i->data==x)
                for(auto j=nodes.begin();j!=nodes.end();++j)
                    if(j->data==y)
                    {
                        i->neighbors.push_back(&(*j));
                        j->neighbors.push_back(&(*i));
                        tmp2.xy.first=&(*i);
                        tmp2.xy.second=&(*j);
                        edges.push_back(tmp2);
                        return;
                    }
    }

    void delete_node(Tipo x)
    {
        for(auto i=nodes.begin();i!=nodes.end();++i)
            if(i->data==x)
            {
                for(auto j=i->neighbors.begin();j!=i->neighbors.end();++j)
                    for(auto k=(*j)->neighbors.begin();k!=(*j)->neighbors.end();++k)
                        if((*k)->data==x)
                        {
                            (*j)->neighbors.erase(k);
                            break;
                        }
                nodes.erase(i);
                break;
            }

        for(auto i=edges.begin();i!=edges.end();)
        {
            if(i->xy.first->data==x || i->xy.second->data==x)
                i=edges.erase(i);
            else
                ++i;
        }
    }

    void delete_edge(Tipo x,Tipo y)
    {
        for(auto i=edges.begin();i!=edges.end();++i)
            if((i->xy.first->data==x && i->xy.second->data==y))
            {
                for(auto j=i->xy.first->neighbors.begin();j!=i->xy.first->neighbors.end();++j)
                    if((*j)->data==y)
                    {
                        i->xy.first->neighbors.erase(j);
                        break;
                    }

                for(auto j=i->xy.second->neighbors.begin();j!=i->xy.second->neighbors.end();++j)
                    if((*j)->data==x)
                    {
                        i->xy.first->neighbors.erase(j);
                        break;
                    }
                edges.erase(i);
                break;
            }
    }

    void merge_graph()
    {
        while(nodes.size()>2)
        {
            int pos=rand()%edges.size();
            auto it=edges.begin();
            advance(it,pos);
            for(auto i=it->xy.first->neighbors.begin();i!=it->xy.first->neighbors.end();++i)
            {
                if((*i)->data != it->xy.second->data)
                    add_edge((*i)->data,it->xy.second->data);
            }
//            cout<<it->xy.first->data<<"  "<<it->xy.second->data<<endl;  La arista que escogio
            delete_node(it->xy.first->data);
        }

        for(auto i=nodes.begin();i!=nodes.end();++i)
            for(auto j=i->neighbors.begin();j!=i->neighbors.end();++j)
                if(i->data==(*j)->data)
                    delete_edge(i->data,(*j)->data);

        cout<<"Nro componentes conexas : "<<edges.size()<<endl;
    }

    void print_graph()
    {
        for(auto i=nodes.begin();i!=nodes.end();++i)
        {
            cout<<"Nodo : "<<i->data<<"  ";
            for(auto j=i->neighbors.begin();j!=i->neighbors.end();++j)
                cout<<"Vecinos : "<<(*j)->data<<"  ";
            cout<<endl;
        }
        cout<<endl;
    }

    void nro_nodes()
    {
        cout<<"NRO NODOS : "<<nodes.size()<<endl;
    }

    void nro_edges()
    {
        cout<<"NRO ARISTAS : "<<edges.size()<<endl;
        for(auto i=edges.begin();i!=edges.end();++i)
            cout<<i->xy.first->data<<"  "<<i->xy.second->data<<endl;
    }

    void Hill_climbing(Tipo inicio,Tipo fin){
        node<Tipo> *temp,*n;
        temp=&(*(nodes.begin()));
        L.push_back(temp);
        list<list<node<Tipo>*>>regreso(1);
        list<node<Tipo>*>temporal;
        while(1){
        if(L.empty()){//Paso 2:Si L es vacio Entonces  la búsqueda no tubo solución
//            cout<<"No hay solucion";
            break;
        }
        else{//Si no entonces  n es el primer nodo  de L
            n=L.front();temporal.assign((*regreso.begin()).begin(),(*regreso.begin()).end());
            if(n->data==fin){//Si n es un nodo objetivo entonces  retornar el camino del nodo inicial hasta n
                cout<<endl<<"Camino :"<<endl;
                for(auto k=temporal.begin();k!=temporal.end();++k)
                    cout<<(*k)->data;
//                retornar camino;
                break;
            }
            else{
            //Paso 3:Sino Remover  n de L
            //Ordenar los hijos de n en orden creciente deacuerdo con las distancias al nodo objetivo
            //Adicionar al inicio de L todos hijos de n, etiquetando  cada uno de
            //ellos con el camino  para el nodo inicial,
            // volver  al paso 2
                temporal.push_front(n);
                L.erase(L.begin());regreso.erase(regreso.begin());
                vector<node<Tipo>*>ordenar;
                  for(unsigned i=0;i<edges.size();i++)
                    if(edges[i].xy.first==n || edges[i].xy.second==n){

                        if(edges[i].xy.first==n)
                            ordenar.push_back(edges[i].xy.second);
                        else
                            ordenar.push_back(edges[i].xy.first);
                    }
                Bubblesort(ordenar);//Ordenar deacuerdo con las distanci al nodo objetivo
                bool esta=false;
                for(auto k=ordenar.begin();k!=ordenar.end();++k){

                    for(auto j=temporal.begin();j!=temporal.end();++j)
                        if(((*k)->data)==(*j)->data)
                            esta=true;
                    if(!esta){
                        L.push_front(*k);
                        regreso.push_front(temporal);}
                    esta=false;
                }
                cout<<"L :"<<endl;
                for(auto i=L.begin();i!=L.end();++i)
                {
                        cout<<"Nodo : "<<(*i)->data<<"  ";

                }
                cout<<endl;

            }
        }
        }

    }


};


int main()
{
    srand(rdtsc());

    graph<char> my_graph;

    my_graph.add_node('S',11);
    my_graph.add_node('A',10.4);
    my_graph.add_node('B',6.9);
    my_graph.add_node('C',4);
    my_graph.add_node('D',8.4);
    my_graph.add_node('E',6.7);
    my_graph.add_node('F',3);
    my_graph.add_node('G',0);


    my_graph.add_edge('S','A',0);
    my_graph.add_edge('A','B',0);
    my_graph.add_edge('B','C',0);
    my_graph.add_edge('S','D',0);
    my_graph.add_edge('D','E',0);
    my_graph.add_edge('E','F',0);
    my_graph.add_edge('F','G',0);
    my_graph.add_edge('A','D',0);
    my_graph.add_edge('B','E',0);
//    cout<<my_graph.nodes[0].data;
//    vector<int> A;
//    A.push_back(3);
//    cout<<A[0];
//    my_graph.print_graph();
    my_graph.Hill_climbing('S','G');
//    list<int>A;
//    A.push_back(3);
//        A.push_back(1);
//    A.push_back(5);
//    A.push_back(2);
//    A.push_back(4);
//    A.push_back(0);
//    A.sort();
//    for(auto i=A.end();i!=A.begin();i--)
//        cout<<(*i);
    return 0;
}

