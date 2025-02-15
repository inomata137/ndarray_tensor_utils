pub fn moveaxis_dyn<S, const NMOVE: usize>(
    a: ndarray::ArrayBase<S, ndarray::IxDyn>,
    source: &[usize; NMOVE],
    destination: &[usize; NMOVE],
) -> ndarray::ArrayBase<S, ndarray::Dim<ndarray::IxDynImpl>>
where
    S: ndarray::Data,
    ndarray::IxDynImpl: ndarray::IntoDimension<Dim = ndarray::IxDyn>,
    ndarray::IxDyn: ndarray::Dimension,
{
    let ndim = a.ndim();
    let dst_src = std::collections::BTreeMap::from_iter(std::iter::zip(destination, source));
    let mut source = *source;
    source.sort();
    let mut rest = (0..ndim).filter(|&i| source.binary_search(&i).is_err());

    let axes = (0..ndim)
        .map(|dst| {
            if let Some(&&src) = dst_src.get(&dst) {
                src
            } else {
                rest.next().unwrap()
            }
        })
        .collect::<Vec<_>>();

    a.permuted_axes(axes)
}

pub fn moveaxis_static<S, const NDIM: usize, const NMOVE: usize>(
    a: ndarray::ArrayBase<S, ndarray::Dim<[usize; NDIM]>>,
    source: &[usize; NMOVE],
    destination: &[usize; NMOVE],
) -> ndarray::ArrayBase<S, ndarray::Dim<[usize; NDIM]>>
where
    S: ndarray::Data,
    [usize; NDIM]: ndarray::IntoDimension<Dim = ndarray::Dim<[usize; NDIM]>>,
    ndarray::Dim<[usize; NDIM]>: ndarray::Dimension,
{
    let dst_src = std::collections::BTreeMap::from_iter(std::iter::zip(destination, source));
    let mut source = *source;
    source.sort();
    let mut rest = (0..NDIM).filter(|&i| source.binary_search(&i).is_err());

    let mut axes = [0; NDIM];
    for (dst, x) in axes.iter_mut().enumerate() {
        *x = if let Some(&&src) = dst_src.get(&dst) {
            src
        } else {
            rest.next().unwrap()
        }
    }

    a.permuted_axes(axes)
}

#[cfg(test)]
mod tests {
    use super::{moveaxis_dyn, moveaxis_static};

    #[test]
    fn test_moveaxis_dyn() {
        let arr_2x3x4x5: ndarray::ArrayBase<
            ndarray::OwnedRepr<f64>,
            ndarray::Dim<ndarray::IxDynImpl>,
        > = ndarray::Array::range(0., (2 * 3 * 4 * 5) as f64, 1.)
            .into_shape_with_order(ndarray::IxDyn(&[2, 3, 4, 5]))
            .unwrap();

        let arr_4x2x3x5 = moveaxis_dyn(arr_2x3x4x5, &[2], &[0]);
        assert_eq!(arr_4x2x3x5.shape(), [4, 2, 3, 5]);
    }

    #[test]
    fn test_moveaxis_static() {
        let arr_2x3x4x5: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 4]>> =
            ndarray::Array::range(0., (2 * 3 * 4 * 5) as f64, 1.)
                .into_shape_with_order(ndarray::Ix4(2, 3, 4, 5))
                .unwrap();

        let arr_4x2x3x5 = moveaxis_static(arr_2x3x4x5, &[2], &[0]);
        assert_eq!(arr_4x2x3x5.shape(), [4, 2, 3, 5]);
    }
}
