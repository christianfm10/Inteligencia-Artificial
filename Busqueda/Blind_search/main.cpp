#include <bits/stdc++.h>
#include <vector>
using namespace std;
#include <iostream>

using namespace std;
template<class T>
class Node{
    public:
    Node(T dat){
        data=dat;
    }
    Node(){}
    T data;
    Node<T> *next;

};
template <class T>
class CList{
    public:
    Node<T> *head=nullptr;
    bool find(T x, Node<T> **&p);
    bool insert(T x);
    bool remove(T x);
    void recorrer();
    bool voltear();
};
template <class T>
void CList<T>::recorrer(){
    Node<T> **p;
    for(p=&head;*p;p=&((*p)->next)){
    cout<<(*p)->data;}
}

template <class T>
bool CList<T>::find(T x, Node<T>**&p){
    for(p=&head;*p && (*p)->data<x;p=&((*p)->next));
    return *p && (*p)->data==x;


}
template<class T>
bool CList<T>::insert(T x){
    Node<T> **p;
    if(find(x,p)) return 0;
    Node<T> *q = new Node<T>(x);
    q->next = *p;
    *p = q;
    return 1;}
template<class T>
bool CList<T>::remove(T x){
    Node<T> **p;
    if(!find(x,p)) return 0;
    Node<T> *q = *p;

    *p=(*p)->next;
    delete q;
//    q->m_next = *p;
//    *p = q;
    return 1;}


int rdtsc()
{
    __asm__ __volatile__("rdtsc");
}

template<class Tipo>
struct node
{
    Tipo data;
    list<node*> neighbors;
    bool verificado=0;
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
    void add_node(Tipo x)
    {
        node<Tipo> tmp;
        tmp.data=x;
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
//    void blind_search(Tipo inicio,Tipo fin){
//        node<Tipo> *temp;
//        temp=&(nodes[0]);
//        L.push_back(temp->data);
//        while(L[0]!=fin){
//            for(int unsigned i=0;i<temp->neighbors.size();i++)
//                L.push_back(temp->data);
//
////        }
//        cout<<L.size();
//        L.erase(L.begin());
//        cout<<L.size();}
////        cout<<temp->data;
////        for(int i=0;)
//
//
//    }
    void blind_search2(Tipo inicio,Tipo fin){
        node<Tipo> *temp,*n;
        temp=&(*(nodes.begin()));
        L.push_back(temp);
        cout<<"qwe"<<temp->data;
        vector<vector<node<Tipo>*>>regreso(nodes.size());
        while(1){
        if(L.empty())
            cout<<"No hay solucion";
        else{
            n=L.front();n->verificado=1;
            if(n->data==fin){
                for(int i=0;i<regreso[0].size();i++)
                    cout<<regreso[0][i]->data;
//                retornar camino;
                cout<<"jasjsa";
                break;
            }
            else{
                L.erase(L.begin());regreso.erase(regreso.begin());
//                for(int unsigned i=0;i<temp->neighbors.size();i++)
//                    L.push_back(temp->neighbors);
                for(int i=0;i<edges.size();i++)
                    if(edges[0]->xy.first){}
                for(auto k=(n)->neighbors.begin();k!=(n)->neighbors.end();++k,i++){
                    if(!((*k)->verificado)){
                        L.push_front(*k);
                        regreso[i].push_back(n);}
                }

            }
        }
        }

    }


};


int main()
{
    srand(rdtsc());

    graph<char> my_graph;

    my_graph.add_node('S');
    my_graph.add_node('A');
    my_graph.add_node('B');
    my_graph.add_node('C');
    my_graph.add_node('D');
    my_graph.add_node('E');
    my_graph.add_node('F');
    my_graph.add_node('G');

    my_graph.add_edge('S','A',3);
    my_graph.add_edge('A','B',4);
    my_graph.add_edge('B','C',4);
    my_graph.add_edge('S','D',4);
    my_graph.add_edge('D','E',2);
    my_graph.add_edge('E','F',4);
    my_graph.add_edge('F','G',3);
    my_graph.add_edge('A','D',5);
    my_graph.add_edge('B','E',5);
//    cout<<my_graph.nodes[0].data;
//    vector<int> A;
//    A.push_back(3);
//    cout<<A[0];
//    my_graph.print_graph();
//    my_graph.blind_search2('S','G');

    return 0;
}
