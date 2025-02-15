pub fn moveaxis<S, I, const NMOVE: usize>(
    a: ndarray::ArrayBase<S, ndarray::Dim<I>>,
    source: &[usize; NMOVE],
    destination: &[usize; NMOVE],
) -> ndarray::ArrayBase<S, ndarray::Dim<I>>
where
    S: ndarray::Data,
    I: ndarray::IntoDimension<Dim = ndarray::Dim<I>> + TryFrom<Vec<usize>, Error: std::fmt::Debug>,
    ndarray::Dim<I>: ndarray::Dimension,
{
    let ndim = a.ndim();
    let dst_src = std::collections::BTreeMap::from_iter(std::iter::zip(destination, source));
    let mut source = *source;
    source.sort();
    let mut rest = (0..ndim).filter(|&i| source.binary_search(&i).is_err());

    let mut axes = Vec::with_capacity(ndim);
    for dst in 0..ndim {
        let axis_idx = if let Some(&&src) = dst_src.get(&dst) {
            src
        } else {
            rest.next().unwrap()
        };
        axes.push(axis_idx);
    }

    let axes = I::try_from(axes).unwrap();
    a.permuted_axes(axes)
}

#[cfg(test)]
mod tests {
    use super::moveaxis;

    #[test]
    fn test_moveaxis_dyn() {
        let arr_2x3x4x5: ndarray::ArrayBase<
            ndarray::OwnedRepr<f64>,
            ndarray::Dim<ndarray::IxDynImpl>,
        > = ndarray::Array::range(0., (2 * 3 * 4 * 5) as f64, 1.)
            .into_shape_with_order(ndarray::IxDyn(&[2, 3, 4, 5]))
            .unwrap();

        let arr_4x2x3x5 = moveaxis(arr_2x3x4x5, &[2], &[0]);

        ndarray::IntoDimension::into_dimension([1, 2, 3, 4]);
        let _: [usize; 4] = TryFrom::try_from(vec![1, 2, 3, 4]).unwrap();
        assert_eq!(arr_4x2x3x5.shape(), [4, 2, 3, 5]);
    }

    #[test]
    fn test_moveaxis_static() {
        let arr_2x3x4x5: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 4]>> =
            ndarray::Array::range(0., (2 * 3 * 4 * 5) as f64, 1.)
                .into_shape_with_order(ndarray::Ix4(2, 3, 4, 5))
                .unwrap();

        let arr_4x2x3x5 = moveaxis(arr_2x3x4x5, &[2], &[0]);
        assert_eq!(arr_4x2x3x5.shape(), [4, 2, 3, 5]);
    }
}
