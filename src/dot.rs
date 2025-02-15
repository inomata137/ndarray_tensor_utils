pub fn tensordot<S1, S2, A, D1, D2, const N: usize>(
    lhs: &ndarray::ArrayBase<S1, D1>,
    rhs: &ndarray::ArrayBase<S2, D2>,
    lhs_axes: &[usize; N],
    rhs_axes: &[usize; N],
) -> std::result::Result<
    ndarray::ArrayBase<ndarray::OwnedRepr<A>, ndarray::IxDyn>,
    ndarray::ShapeError,
>
where
    S1: ndarray::Data<Elem = A>,
    S2: ndarray::Data<Elem = A>,
    A: ndarray::LinalgScalar,
    D1: ndarray::Dimension,
    D2: ndarray::Dimension,
{
    let lhs_axes_uniq = std::collections::BTreeSet::from_iter(lhs_axes);
    assert_eq!(lhs_axes_uniq.len(), N, "lhs_axes has duplicate entries");
    let rhs_axes_uniq = std::collections::BTreeSet::from_iter(rhs_axes);
    assert_eq!(rhs_axes_uniq.len(), N, "rhs_axes has duplicate entries");

    let lhs_permutation = (0..lhs.ndim())
        .filter(|ax| !lhs_axes_uniq.contains(ax))
        .chain(lhs_axes.iter().copied())
        .collect::<Vec<_>>();
    let rhs_permutation = rhs_axes
        .iter()
        .copied()
        .chain((0..rhs.ndim()).filter(|ax| !rhs_axes_uniq.contains(ax)))
        .collect::<Vec<_>>();

    let lhs_permuted = lhs.view().into_dyn().permuted_axes(lhs_permutation);
    let (out_left_shape, dot_shape) = lhs_permuted.shape().split_at(lhs.ndim() - N);
    let out_left_size = out_left_shape.iter().product::<usize>();
    let dot_size = dot_shape.iter().product::<usize>();

    let rhs_permuted = rhs.view().into_dyn().permuted_axes(rhs_permutation);
    let (dot_shape, out_right_shape) = rhs_permuted.shape().split_at(N);
    assert_eq!(dot_size, dot_shape.iter().product::<usize>());
    let out_right_size = out_right_shape.iter().product::<usize>();

    let out_shape = out_left_shape
        .iter()
        .chain(out_right_shape)
        .copied()
        .collect::<Vec<_>>();

    lhs_permuted
        .to_shape([out_left_size, dot_size])?
        .dot(&rhs_permuted.to_shape([dot_size, out_right_size])?)
        .into_shape_with_order(out_shape)
}

pub fn outer_product<S1, S2, A, D1, D2>(
    lhs: &ndarray::ArrayBase<S1, D1>,
    rhs: &ndarray::ArrayBase<S2, D2>,
) -> std::result::Result<
    ndarray::ArrayBase<ndarray::OwnedRepr<A>, ndarray::IxDyn>,
    ndarray::ShapeError,
>
where
    S1: ndarray::Data<Elem = A>,
    S2: ndarray::Data<Elem = A>,
    A: ndarray::LinalgScalar,
    D1: ndarray::Dimension,
    D2: ndarray::Dimension,
{
    let mut lhs_shape = Vec::from(lhs.shape());
    lhs_shape.push(1);
    let lhs = lhs
        .view()
        .into_shape_with_order(ndarray::IxDyn(&lhs_shape))
        .unwrap();

    let mut rhs_shape = vec![1];
    rhs_shape.extend(rhs.shape());
    let rhs = rhs
        .view()
        .into_shape_with_order(ndarray::IxDyn(&rhs_shape))
        .unwrap();
    tensordot(&lhs, &rhs, &[lhs.ndim() - 1], &[0])
}

#[cfg(test)]
mod tests {
    use super::{outer_product, tensordot};

    #[test]
    fn test_tensordot() {
        let arr_2x3x4 = ndarray::array![
            [[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]],
            [
                [13., 14., 15., 16.],
                [17., 18., 19., 20.],
                [21., 22., 23., 24.]
            ]
        ];
        let arr_4x2x3 = ndarray::array![
            [[1., 2., 3.], [4., 5., 6.]],
            [[7., 8., 9.], [10., 11., 12.]],
            [[13., 14., 15.], [16., 17., 18.]],
            [[19., 20., 21.], [22., 23., 24.]]
        ];

        let result = tensordot(&arr_2x3x4, &arr_4x2x3, &[0, 2], &[1, 0]);

        assert!(result.is_ok());

        assert_eq!(
            result.unwrap(),
            ndarray::array![
                [914., 982., 1050.],
                [1282., 1382., 1482.],
                [1650., 1782., 1914.],
            ]
            .into_dyn() // shape: (3, 3)
        );
    }

    #[test]
    fn test_outer_product() {
        let arr_1x2 = ndarray::array![[1., 2.],];
        let arr_3x4 = ndarray::array![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.],];

        let result = outer_product(&arr_1x2, &arr_3x4);

        assert!(result.is_ok());

        assert_eq!(
            result.unwrap(),
            ndarray::array![[
                [[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.],],
                [[2., 4., 6., 8.], [10., 12., 14., 16.], [18., 20., 22., 24.],],
            ],]
            .into_dyn() // shape: (1, 2, 3, 4)
        );
    }
}
