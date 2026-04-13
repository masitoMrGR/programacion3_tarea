#include <iostream>
#include <vector>
#include <cstdlib> 
#include <ctime>   
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <cmath>
using namespace std;
class TensorTransform;
class Tensor {
    private:
        vector<size_t> forma;
        int total=1;
        double* valores;
        bool owner = true;
        Tensor(const vector<size_t>& shape, double* data, int tot, bool data_dueñ)
            : forma(shape), total(tot), valores(data), owner(data_dueñ) {}
    
    public:
        friend class ReLU;
        friend class Sigmoid;
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
            owner = true;
        };
        
        friend ostream& operator<<(ostream& os, const Tensor& t){
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
            long unsigned int n = maxi-mino;
            
            vector<size_t> shape = {n};
            vector <double> values (n);
            
            for (int i = 0; i<n;++i){
                values[i] = mino+i;
            }
            return Tensor (shape, values);
        };
        
        //Constructor de copia
        Tensor (const Tensor & other){
            forma = other.forma;
            total=other.total;
            valores = new double[total];
            for (int i = 0; i<total;++i){
                valores[i] = other.valores[i];  
            };
            owner = true;
        };
        //Constructor de movimiento
        Tensor (Tensor && other ) noexcept{
            forma = move(other.forma);
            total= other.total;
            valores = other.valores;
            other.valores = nullptr;
            other.forma = {};
            other.total=0;
            owner = other.owner;
            other.owner = false;
        };
        //Asignacion de copia
        Tensor& operator=(const Tensor& other){
    if (this != &other){
        double* newData = new double[other.total];
        std::copy(other.valores, other.valores + other.total, newData);

        if (owner && valores != nullptr){
            delete[] valores;
        }

        valores = newData;
        total = other.total;
        forma = other.forma;
        owner = true;
    }
    return *this;
}
        //Asignacion de movimiento
        Tensor& operator=(Tensor&& other) noexcept{
    if (this != &other){
        if (owner && valores != nullptr){
            delete[] valores;
        }

        valores = other.valores;
        total = other.total;
        forma = move(other.forma);
        owner = other.owner;

        other.valores = nullptr;
        other.total = 0;
        other.forma = {};
        other.owner = false;
    }
    return *this;
}

        Tensor view(const vector<size_t>& new_shape) const;
        Tensor unsqueeze(size_t dim) const;

        static Tensor concat(const vector<Tensor>& tensors, size_t dim);

        friend Tensor operator+(const Tensor& a, const Tensor& b);
        friend Tensor operator-(const Tensor& a, const Tensor& b);
        friend Tensor operator*(const Tensor& a, const Tensor& b);
        friend Tensor operator*(const Tensor& a, double scalar);
        friend Tensor operator*(double scalar, const Tensor& a);

        friend Tensor dot(const Tensor& a, const Tensor& b);
        friend Tensor matmul(const Tensor& a, const Tensor& b);
        
        Tensor apply(const TensorTransform& transform) const;
        
        ~Tensor(){
            if (owner and valores != nullptr){
                delete[] valores;
            }
        };
};


class TensorTransform {
    public:
        virtual Tensor apply(const Tensor& t) const = 0;
        virtual ~TensorTransform() = default;
};

class ReLU : public TensorTransform {
    public:
        Tensor apply(const Tensor& t) const override;
};

class Sigmoid : public TensorTransform {
    public:
        Tensor apply(const Tensor& t) const override;
};




Tensor Tensor::apply(const TensorTransform& transform) const {
    return transform.apply(*this);
};

Tensor ReLU::apply(const Tensor& t) const {
    vector<double> values(t.total);

    for (int i = 0; i < t.total; ++i) {
        values[i] = max(0.0, t.valores[i]);
    }

    return Tensor(t.forma, values);
};

Tensor Sigmoid::apply(const Tensor& t) const {
    vector<double> values(t.total);

    for (int i = 0; i < t.total; ++i) {
        values[i] = 1.0 / (1.0 + exp(-t.valores[i]));
    }

    return Tensor(t.forma, values);
};

Tensor operator+(const Tensor& a, const Tensor& b) {
    if (a.forma != b.forma) {
        throw invalid_argument("Las dimensiones no son compatibles para la suma entre los tensores");
    };

    vector<double> values(a.total);
    for (int i = 0; i < a.total; ++i) {
        values[i] = a.valores[i] + b.valores[i];
    };

    return Tensor(a.forma, values);
}

Tensor operator-(const Tensor& a, const Tensor& b) {
    if (a.forma != b.forma) {
        throw invalid_argument("Las dimensiones no son compatibles para la resta");
    };

    vector<double> values(a.total);
    for (int i = 0; i < a.total; ++i) {
        values[i] = a.valores[i] - b.valores[i];
    };

    return Tensor(a.forma, values);
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    if (a.forma != b.forma) {
        throw invalid_argument("Las dimensiones no son compatibles para la multiplicacion elemento a elemento");
    };

    vector<double> values(a.total);
    for (int i = 0; i < a.total; ++i) {
        values[i] = a.valores[i] * b.valores[i];
    };

    return Tensor(a.forma, values);
}

Tensor operator*(const Tensor& a, double scalar) {
    vector<double> values(a.total);
    for (int i = 0; i < a.total; ++i) {
        values[i] = a.valores[i] * scalar;
    };

    return Tensor(a.forma, values);
}

Tensor operator*(double scalar, const Tensor& a) {
    return a * scalar;
}

Tensor Tensor::view(const vector<size_t>& new_shape) const {
    if (new_shape.empty() || new_shape.size() > 3) {
        throw invalid_argument("La nueva forma debe tener entre 1 y 3 dimensiones");
    }

    int new_total = 1;
    for (size_t s : new_shape) {
        if (s == 0) {
            throw invalid_argument("Las dimensiones no pueden ser 0");
        }
        new_total *= s;
    }

    if (new_total != total) {
        throw invalid_argument("El total de elementos no coincide para view");
    }

    return Tensor(new_shape, valores, total, false);
}

Tensor Tensor::unsqueeze(size_t dim) const {
    if (forma.size() >= 3) {
        throw invalid_argument("Bro, no se puede exceder de 3 dimensiones");
    };

    if (dim > forma.size()) {
        throw invalid_argument("La posicion para unsqueeze es invalida");
    };

    vector<size_t> new_shape = forma;
    new_shape.insert(new_shape.begin() + dim, 1);

    return Tensor(new_shape, valores, total, false);
}

//No hay acaso otra forma de hacer esta parte?
Tensor Tensor::concat(const vector<Tensor>& tensors, size_t dim) {
    if (tensors.empty()) {
        throw invalid_argument("No se puede concatenar un vector vacio de tensores");
    }

    size_t ndims = tensors[0].forma.size();

    if (ndims == 0 || ndims > 3) {
        throw invalid_argument("Cantidad de dimensiones invalida");
    }

    if (dim >= ndims) {
        throw invalid_argument("Dimension de concatenacion invalida");
    }

    vector<size_t> new_shape = tensors[0].forma;
    new_shape[dim] = 0;

    for (const auto& t : tensors) {
        if (t.forma.size() != ndims) {
            throw invalid_argument("Todos los tensores deben tener la misma cantidad de dimensiones");
        }

        for (size_t i = 0; i < ndims; ++i) {
            if (i != dim && t.forma[i] != tensors[0].forma[i]) {
                throw invalid_argument("Las dimensiones no son compatibles para concat");
            }
        }

        new_shape[dim] += t.forma[dim];
    }

    int new_total = 1;
    for (size_t s : new_shape) {
        new_total *= s;
    }

    double* new_data = new double[new_total];

    if (ndims == 1) {
        int pos = 0;
        for (const auto& t : tensors) {
            for (int i = 0; i < t.total; ++i) {
                new_data[pos++] = t.valores[i];
            }
        }
    }
    else if (ndims == 2) {
        size_t rows = new_shape[0];
        size_t cols = new_shape[1];

        if (dim == 0) {
            int pos = 0;
            for (const auto& t : tensors) {
                for (int i = 0; i < t.total; ++i) {
                    new_data[pos++] = t.valores[i];
                }
            }
        } else if (dim == 1) {
            size_t current_col_offset = 0;
            for (size_t r = 0; r < rows; ++r) {
                size_t out_col = 0;
                for (const auto& t : tensors) {
                    for (size_t c = 0; c < t.forma[1]; ++c) {
                        new_data[r * cols + out_col] = t.valores[r * t.forma[1] + c];
                        ++out_col;
                    }
                }
            }
        }
    }
    else if (ndims == 3) {
        size_t d0 = new_shape[0];
        size_t d1 = new_shape[1];
        size_t d2 = new_shape[2];

        if (dim == 0) {
            int pos = 0;
            for (const auto& t : tensors) {
                for (int i = 0; i < t.total; ++i) {
                    new_data[pos++] = t.valores[i];
                }
            }
        }
        else if (dim == 1) {
            for (size_t i = 0; i < d0; ++i) {
                size_t out_j = 0;
                for (const auto& t : tensors) {
                    for (size_t j = 0; j < t.forma[1]; ++j) {
                        for (size_t k = 0; k < d2; ++k) {
                            new_data[i * d1 * d2 + out_j * d2 + k] =
                                t.valores[i * t.forma[1] * t.forma[2] + j * t.forma[2] + k];
                        }
                        ++out_j;
                    }
                }
            }
        }
        else if (dim == 2) {
            for (size_t i = 0; i < d0; ++i) {
                for (size_t j = 0; j < d1; ++j) {
                    size_t out_k = 0;
                    for (const auto& t : tensors) {
                        for (size_t k = 0; k < t.forma[2]; ++k) {
                            new_data[i * d1 * d2 + j * d2 + out_k] =
                                t.valores[i * t.forma[1] * t.forma[2] + j * t.forma[2] + k];
                            ++out_k;
                        }
                    }
                }
            }
        }
    }

    return Tensor(new_shape, new_data, new_total, true);
}
//me voy a dormir bro

Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.forma != b.forma) {
        throw invalid_argument("Las dimensiones no son compatibles para dot");
    }

    double suma = 0.0;
    for (int i = 0; i < a.total; ++i) {
        suma += a.valores[i] * b.valores[i];
    }

    return Tensor({1}, {suma});
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.forma.size() != 2 || b.forma.size() != 2) {
        throw invalid_argument("matmul solo se permite para tensores 2D");
    }

    if (a.forma[1] != b.forma[0]) {
        throw invalid_argument("Las dimensiones no son compatibles para matmul");
    }

    size_t filas = a.forma[0];
    size_t comun = a.forma[1];
    size_t columnas = b.forma[1];

    vector<double> values(filas * columnas, 0.0);

    for (size_t i = 0; i < filas; ++i) {
        for (size_t j = 0; j < columnas; ++j) {
            double suma = 0.0;
            for (size_t k = 0; k < comun; ++k) {
                suma += a.valores[i * comun + k] * b.valores[k * columnas + j];
            }
            values[i * columnas + j] = suma;
        }
    }

    return Tensor({filas, columnas}, values);
}

int main() {
    srand(time(NULL));

    Tensor datos = Tensor::random({1000, 20, 20}, 0.0, 1.0);

    Tensor X = datos.view({1000, 400});

    Tensor pesos1 = Tensor::random({400, 100}, -1.0, 1.0);
    Tensor salida1 = matmul(X, pesos1);

    Tensor ajuste1 = Tensor::random({1, 100}, -1.0, 1.0);
    vector<Tensor> lista_ajuste1(1000, ajuste1);
    Tensor ajuste1_rep = Tensor::concat(lista_ajuste1, 0);

    Tensor salida1_ajustada = salida1 + ajuste1_rep;

    ReLU relu;
    Tensor activacion1 = salida1_ajustada.apply(relu);

    Tensor pesos2 = Tensor::random({100, 10}, -1.0, 1.0);
    Tensor salida2 = matmul(activacion1, pesos2);

    Tensor ajuste2 = Tensor::random({1, 10}, -1.0, 1.0);
    vector<Tensor> lista_ajuste2(1000, ajuste2);
    Tensor ajuste2_rep = Tensor::concat(lista_ajuste2, 0);

    Tensor salida2_ajustada = salida2 + ajuste2_rep;

    Sigmoid sig;
    Tensor resultado = salida2_ajustada.apply(sig);

    cout << "Resultado final:" << endl;
    cout << resultado << endl;

    return 0;
}