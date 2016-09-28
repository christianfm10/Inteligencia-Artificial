#include <iostream>
#include <bits/stdc++.h>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
using namespace std;
template<class Tipo>
struct node
{
    Tipo data;
    list<node*> neighbors;
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
    vector<node<Tipo>> nodes;
    vector<edge<Tipo>> edges;
    vector<node<Tipo>> cromosomas;
    int poblacion;
    int generaciones;
    graph(int x,int y){
        poblacion=x;
        generaciones=y;
    }
    void add_node(Tipo x)
    {
        node<Tipo> tmp;
        tmp.data=x;
        nodes.push_back(tmp);
        cromosomas.assign(nodes.begin(),nodes.end());
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
//Alg Genetico aplicado al Problema del viajero
    void Alg_Genetico(){

            int n=nodes.size()-1;
            srand(time(NULL));
            vector<vector<node<Tipo>*>> Padres;
            Padres.resize(poblacion);
            vector<vector<node<Tipo>*>> Hijos;vector<node<Tipo>*> OPOP;int OP;
            Hijos.resize(poblacion);
            for(unsigned i=0;i<Padres.size();i++)
                Padres[i].resize(n);
            GenerarPadres(Padres);
            for(unsigned i=0;i<Hijos.size();i++)
                Hijos[i].resize(n);
            for(int cont=0;cont<generaciones;cont++){
                for(unsigned i=0;i<Padres.size();i=i+2){
                    cruzamiento(Padres,Hijos,i);


                }
                for(unsigned i=0;i<Hijos.size();i++)
                    mutacion(Hijos[i]);
                int pos;
                vector<int> listcalidad;
                for(unsigned i=0;i<Hijos.size();i++)
                    listcalidad.push_back(aptitud(Hijos[i]));
                cout<<endl<<"El mejor de la generacion "<< cont <<" es:"<<endl;
                int mejor=elmejordesugeneracion(listcalidad,pos);
                for(unsigned i=0;i<Hijos[pos].size();i++)
                    cout<<Hijos[pos][i]->data;
                cout<<endl<<mejor<<endl;
                if(mejor<OP)
                {   OP=mejor;
                    OPOP.assign(Hijos[pos].begin(),Hijos[pos].end());
                }

                Padres.assign(Hijos.begin(),Hijos.end());
                Padres[0].assign(OPOP.begin(),OPOP.end());
            }
            cout<<endl<<"El mejor de todos es:"<<endl;
            for(unsigned i=0;i<OPOP.size();i++)
                    cout<<OPOP[i]->data;
            cout<<endl<<"Distancia :"<<OP;
            return;
        }
    void GenerarPadres(vector<vector<node<Tipo>*>> &Padres){
        string A,B;
        for(int i=0;i<nodes.size()-1;i++)
        {
            A.push_back((char)(49+i));
        }
       int pos,lugar;
    for(unsigned j=0;j<Padres.size();j++){
        B=A;
        for(unsigned i=0; i<nodes.size()-1;i++){
                lugar=rand()%B.size();
                pos=(int)B[lugar];
                B.erase(lugar,1);

                Padres[j][i] = &nodes[pos-48];
        }
    }
    }
    int elmejordesugeneracion(vector<int> A,int &pos){
        int mejor=10000;
        for(unsigned i=0;i < A.size();i++)
            if(mejor > A[i]){
                mejor = A[i];
                pos = i;
            }
        return mejor;
    }
    int elpeordesugeneracion(vector<int> A){
        int peor=0;
        for(unsigned i=0;i < A.size();i++)
            if(peor < A[i])
                peor=A[i];
        return peor;
    }
    int aptitud(vector<node<Tipo>*> A){
           int aptidud=0;
           for(unsigned j=1;j<A.size();j++){

                for(auto i=edges.begin();i!=edges.end();++i)
                {
                    if(i->xy.first->data == A[j]->data || i->xy.first->data==A[j-1]->data)
                        if(i->xy.second->data==A[j-1]->data || i->xy.second->data==A[j]->data){
                            aptidud += i->weight;
                            break;
                        }
                }
           }

            for(auto i=edges.begin();i!=edges.end();++i)
            {
                if(i->xy.first->data == nodes[0].data || i->xy.first->data == A[0]->data)
                    if(i->xy.second->data==nodes[0].data || i->xy.second->data==A[0]->data){
                        aptidud += i->weight;
                    }
                if(i->xy.first->data == nodes[0].data || i->xy.first->data==A[A.size()-1]->data)
                    if(i->xy.second->data==nodes[0].data || i->xy.second->data==A[A.size()-1]->data){
                        aptidud += i->weight;
                    }
            }
           return aptidud;
    }
    //Mutacion basada en posicion
    void mutacion(vector<node<Tipo>*> &A){
        int a,b;
        a=rand()%(nodes.size()-1);
        b=rand()%(nodes.size()-1);
        node<Tipo> *aux;
        while(1){
            if(a!=b){
                aux=A[a];
                A[a]=A[b];
                A[b]=aux;
                break;}
    //        a=rand()%nodes.size();
            b=rand()%(nodes.size()-1);
        }



    }
    //CrossOver De orden
    void cruzamiento(vector<vector<node<Tipo>*>>&A,vector<vector<node<Tipo>*>> &B,int i){

            vector<int> vectbits;
            vectbits.resize(A.size());
            producir_bits(vectbits);
            B[i].assign(A[i+1].begin(),A[i+1].end());
            B[i+1].assign(A[i].begin(),A[i].end());
            cout<<endl;
            for(unsigned j=0;j<vectbits.size();j++){
                    if(vectbits[j]==0){
                        B[i][j]=NULL;
                        B[i+1][j]=NULL;
                    }
            }

            for(unsigned j=0;j<B[i].size();j++){
                if(B[i][j]==NULL){
                    for(unsigned k=0;k<A[i].size();k++){
                        unsigned cont=0;
                        for(unsigned l=0;l<A[i].size();l++,cont++){
                            if(A[i][k]==B[i][l])
                                break;

                        }
                        if(cont==A[i].size()){
                            B[i][j]=A[i][k];
                        }
                    }
                }
            }
            for(unsigned j=0;j<B[i+1].size();j++){
                if(B[i+1][j]==NULL){
                    for(unsigned k=0;k<A[i+1].size();k++){
                        unsigned cont=0;
                        for(unsigned l=0;l<A[i+1].size();l++,cont++){
                            if(A[i+1][k]==B[i+1][l])
                                break;

                        }
                        if(cont==A[i+1].size()){
                            B[i+1][j]=A[i+1][k];
                        }
                    }
                }
            }
    }
//Genera vector con numeros binarios
    void producir_bits(vector<int> &vectbits){
        for(unsigned i=0;i<vectbits.size();i++)
            vectbits[i]=rand()%2;
    }



    };
void NRecorrerVector(vector<node<char>*> A){
    for(unsigned i=0;i<A.size();i++)
        cout<<A[i]->data;
    }

int main()
{


    int poblacion=4;
    int generaciones=40;
    graph<char> my_graph(poblacion,generaciones);

    my_graph.add_node('A');
    my_graph.add_node('B');
    my_graph.add_node('C');
    my_graph.add_node('D');
    my_graph.add_node('E');
    my_graph.add_node('F');
    my_graph.add_node('G');
    my_graph.add_edge('A','B',3);
    my_graph.add_edge('A','C',1);
    my_graph.add_edge('A','D',5);
    my_graph.add_edge('A','E',2);
    my_graph.add_edge('A','F',8);
    my_graph.add_edge('A','G',1);
    my_graph.add_edge('B','C',7);
    my_graph.add_edge('B','D',1);
    my_graph.add_edge('B','E',3);
    my_graph.add_edge('B','F',1);
    my_graph.add_edge('B','G',9);
    my_graph.add_edge('C','D',6);
    my_graph.add_edge('C','E',1);
    my_graph.add_edge('C','F',7);
    my_graph.add_edge('C','G',5);
    my_graph.add_edge('D','E',4);
    my_graph.add_edge('D','F',3);
    my_graph.add_edge('D','G',1);
    my_graph.add_edge('E','F',1);
    my_graph.add_edge('E','G',6);
    my_graph.add_edge('F','G',5);
    my_graph.Alg_Genetico();

     return 0;
}
