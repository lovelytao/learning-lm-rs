use std::{slice, sync::Arc, vec};
pub struct Tensor<T> {
    data: Arc<Box<[T]>>,   // data of the tensor
    shape: Vec<usize>,     // shape of the tensor
    offset: usize,         // offset of the tensor
    length: usize,         // length of the tensor   
}

impl<T: Copy + Clone + Default> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        Tensor {
            // 这句代码将一个 Vec<T> 转换成了一个 Box<[T]>（堆分配的切片），
            // 然后尝试将其转换成所需的类型（通过 try_into()），
            // 接着用 Arc 进行包装，以便在多线程环境中共享。
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            offset: 0,
            length: length,
        }
    }

    pub fn default(shape: &Vec<usize>) -> Self {
        // 根据shape计算出data的长度
        let length = shape.iter().product();
        // 利用T的默认值构建一个长度为length的Vec
        let data = vec![T::default(); length];
        // 调用new函数构建一个Tensor
        Self::new(data, shape)
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
        }
    }


}

// Some helper functions for testing and debugging
impl Tensor<f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();
        
        return a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel));
    }
    #[allow(unused)]
    pub fn print(&self){
        println!("shpae: {:?}, offset: {}, length: {}", self.shape, self.offset, self.length);
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}
