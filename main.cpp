#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <sys/time.h>
using namespace std;
int init_numpy(){//初始化 numpy 执行环境，主要是导入包，python3.0以上用int返回类型

    import_array();
}
PyObject* load_model()
{
    //调用python函数，提前单次获取模型实例，驻于内存，加速推理时间
    //需要将python中某些变量在python脚本执行结束后仍然保存在内存中，可以用函数返回值返回出来
    PyObject *pModule  = nullptr;
    PyObject *pDict    = nullptr;

    pModule  = PyImport_ImportModule("final");
    pDict    = PyModule_GetDict(pModule);
    PyObject *pFunc = PyDict_GetItemString(pDict, "load_models_in_disk");
    PyObject *model_list = PyObject_CallObject(pFunc, nullptr);
    Py_DECREF(pModule);
    Py_DECREF(pDict);
    cout<<"load model finish"<<endl;
    return model_list;
}
void prob(PyObject *model_list){
    PyObject *pModule  = nullptr;
    PyObject *pDict    = nullptr;

    pModule  = PyImport_ImportModule("model");//对应python文件名
    pDict    = PyModule_GetDict(pModule);

    PyObject *pFuncMain = PyDict_GetItemString(pDict, "predict_proba_");//传入函数名
    double CArrays[8][17] =  {
        { 1.57749969,0.00000000,-7.44845000, 6.30254000, -4.18040000, -1.98820000, -9.52929000e-01, -8.19152295e+03,  2.32128198e+03  ,1.53430703e+04, -2.18869950e+01 ,-2.16578120e+01,  1.02903259e+02 , 6.28000000e-01 ,-2.70000000e-02 , 2.61000000e-01, -7.32000000e-01},
        { 1.57749969, 5.72960000,-2.29183200 , 6.30254000, -1.62596000, -1.26124000, -1.35753700e+00, -8.18820020e+03,  2.32632397e+03 , 1.52760342e+04 ,-2.19442900e+01 ,-2.16005170e+01,  1.02960548e+02  ,6.28000000e-01, -2.70000000e-02  ,2.61000000e-01, -7.32000000e-01},
        { 1.57749969,-5.15662000, -7.44845000 , 1.60428200,  4.14050000 ,-9.86220000 ,-8.06675000e-01, -8.19656543e+03,  2.30283276e+03  ,1.52995254e+04, -2.21161790e+01, -2.17151070e+01,  1.03017845e+02 , 6.27000000e-01 ,-2.70000000e-02 , 2.62000000e-01, -7.32000000e-01},
        { 1.57749969, -8.02141000 , -1.31780300, -2.75019800, -9.81230000, -6.81520000 ,-1.27899000e+00, -8.22836426e+03,  2.34139282e+03,  1.52944834e+04 ,-2.21161790e+01 ,-2.16578120e+01,  1.03075142e+02,  6.27000000e-01, -2.70000000e-02,  2.62000000e-01, -7.32000000e-01},
        { 1.57749969, 4.01071000,  -1.31780300 , 8.02141000, -1.45568000, -2.90620000e-02 ,-1.32350000e+00, -0.20997266e+03,  2.32632397e+03 , 1.53011865e+04 ,-2.21161790e+01 ,-2.16578120e+01,  1.03132439e+02  ,6.27000000e-01, -2.70000000e-02,  2.62000000e-01, -7.32000000e-01},
        { 1.57749969, 6.30254000,  -2.11994500,  1.43239500,  1.12820000,  3.03290000e-02, -9.45294000e-01 ,-8.17645410e+03,  2.34809644e+03 , 1.52911602e+04 ,-2.21161790e+01, -2.16578120e+01,  1.03189735e+02 , 6.27000000e-01, -2.70000000e-02 , 2.62000000e-01, -7.32000000e-01},
        { 1.57749969, -5.72958000,  1.14591600, -2.00535300, -5.33370000,  3.56260000e-02, -1.13160600e+00, -8.18653857e+03,  2.33130884e+03,  1.53463936e+04, -2.21161790e+01, -2.16578120e+01,  1.03189735e+02,  6.26000000e-01, -2.70000000e-02, 2.62000000e-01, -7.33000000e-01},
        { 1.57749969, -1.14592000 ,-1.08862000,  1.43239500, -1.28929000, -2.76340000e-02, -1.33114600e+00, -8.20831055e+03,  2.31962036e+03,  1.52760342e+04, -2.21161790e+01, -2.16578120e+01,  1.03189735e+02 , 6.26000000e-01 ,-2.70000000e-02 , 2.62000000e-01, -7.33000000e-01},
    };
    npy_intp Dims[2]={8,17};

    PyObject *PyArray  = PyArray_SimpleNewFromData(2, Dims, NPY_DOUBLE, CArrays); //生成包含这个多维数组的PyObject对象，使用PyArray_SimpleNewFromData函数，第一个参数2表示维度，第二个为维度数组Dims,第三个参数指出数组的类型，第四个参数为数组
    PyObject *ArgArray = PyTuple_New(2);//函数参数数组，对应python函数有多少参数
    PyTuple_SetItem(ArgArray, 0, PyArray);
    PyTuple_SetItem(ArgArray, 1, model_list);
    timeval start,end;
    gettimeofday(&start,0);
    int Index_i = 0, Index_k = 0, size = 1000;
    //模拟1000次调用
    for(Index_i=0;Index_i<size;Index_i++)
    {

        PyObject *class_prob = PyObject_CallObject(pFuncMain, ArgArray);
        Py_DECREF(class_prob);
    }
    gettimeofday(&end,0);
    double timeuse = 1000000*(end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    timeuse/=1000;//ms
    cout<<"time use is :"<<timeuse<<"ms"<<endl;
    Py_DECREF(pModule);
    Py_DECREF(pDict);

    Py_DECREF(PyArray);
    Py_DECREF(ArgArray);

}
int main()
{
    //引入python环境
    Py_Initialize();
    init_numpy();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('your python script path')");
    PyObject *model_list = load_model();
    prob(model_list);
    Py_DECREF(model_list);
    return 0;
}