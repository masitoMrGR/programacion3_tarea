#include <iostream>
#include <vector>
#include <cstdlib> 
#include <ctime>   
using namespace std;

class Tensor {
    vector<size_t> forma;
    int total=1;
    double* valores;
    
    public:
        Tensor (const vector <size_t>& shape , const vector<double>& values){
            total=1;
            for (size_t s : shape) {
                if (s == 0) {
                    throw invalid_argument("Dimensiones no pueden ser 0");
                };
            };
            
            for (size_t s : shape) {
                total *= s;
            };

            if (values.size() != total) {
                throw invalid_argument("La cantidad de valores no coinciden con las dimensiones que ha ingresado");
            };
            
            valores = new double [total];
            
            for (int i =0; i<total;++i){
                valores[i]=values[i];
            };   
            forma = shape;
        };
        
        friend ostream& operator<<(ostream& os, Tensor& t){
            if (t.forma.size () == 1){
                os<<"[ ";
                for (int i=0; i<t.total;++i){
                    os<<t.valores[i]<<" ";
                };
                os<<"]"<<endl;
            } else if(t.forma.size()==2){
                for (int i=0; i<t.forma[0];++i){
                    os<<"[ ";
                    for (int j=0; j<t.forma[1];++j){
                        os<<t.valores[i*t.forma[1]+j]<<" ";
                    };
                    os<<"]"<<endl;
                };
            }
            else if (t.forma.size()==3){
                int c=1;
                for (int i=0; i<t.forma[0];++i){
                    os<<"Capa: "<<c<<endl;
                    for (int j = 0; j<t.forma[1];++j){
                        os<<"[ ";
                        for (int k = 0; k<t.forma[2];++k){
                            os<<t.valores[i*t.forma[1]*t.forma[2]+j*t.forma[2]+k]<<" ";
                        }
                        os<<"]"<<endl;
                    };
                    c++;
                };
            }
            
            return os;
        };
        
        static Tensor zeros(const vector<size_t>&shape){
            int tot =1;
            for (size_t s : shape) {
                if (s==0) {
                    throw invalid_argument("Dimensiones no pueden ser 0");
                }
                tot *= s;
            };
            vector<double> values (tot, 0);
            return Tensor (shape,values);
        };
        
        static Tensor ones(const vector<size_t>&shape){
            int tot =1;
            for (size_t s : shape) {
                if (s==0) {
                    throw invalid_argument("Dimensiones no pueden ser 0");
                }
                tot *= s;
            };
            vector<double> values (tot, 1);
            return Tensor (shape,values);
        };
        static Tensor random(const vector<size_t>&shape, double mino, double maxi){
            int tot =1;
            for (size_t s : shape) {
                if (s==0) {
                    throw invalid_argument("Dimensiones no pueden ser 0");
                };
                tot *= s;
            };
            vector<double> values (tot);
            for (size_t i = 0; i < tot; ++i) {
                values[i] = mino + (rand() % (int)((maxi - mino)*100)) / 100.0;
            };
            return Tensor (shape,values);
        };
        
        static Tensor arange(int mino, int maxi){
            int n = maxi-mino;
            
            vector<size_t> shape = {n};
            vector <double> values (n);
            
            for (int i = 0; i<n;++i){
                values[i] = mino+i;
            }
            return Tensor (shape, values);
            
            
        };
        
        
        
        ~Tensor(){
                delete[] valores;    
            };
};





int main()

{
    srand(time(NULL));
    vector<size_t> form ={2,3};
    vector<double> val={4,5,6,7,8,9};

    Tensor t1 ({2,3,2}, {1,2,3,4,5,6,7,8,9,10,11,12});
    
    cout<<t1;
}