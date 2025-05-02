#ifndef _PERIODIZE_HPP_
#define _PERIODIZE_HPP_

#include <csbq.hpp>

/**
 * Gives the far-field component to periodize Stokes flow.
 */
template <class Real> class Periodize {
  //using PrecompReal = long double;
  using PrecompReal = double;

  public:

  /**
   * @return Coordinates (in AoS order) of proxy points around unit box [0,1]^3.
   */
  static const sctl::Vector<Real>& GetProxySurf();

  /**
   * Return the far-field requires to periodize the flow.
   *
   * @param[out] U_far the far-field potential evaluated at Xt.
   *
   * @param[in] Xt the target coordinates in the unit box [0,1]^3
   *
   * @param[in] U_proxy the outgoing field from the unit box evaluated at the proxy points.
   */
  static void EvalFarField(sctl::Vector<Real>& U_far, const sctl::Vector<Real>& Xt, const sctl::Vector<Real>& U_proxy);

  private:

  // Notation:
  // UC: upward-check (outgoing field evaluated at the upward-check points)
  // UE: upward-equivalent (equivalent density at the proxy points that produces the same outgoing field)
  // DC: downward-check (incoming field evaluated at the downward-check points)
  // DE: downward-equivalent (equivalent density at the proxy points that produces the same incoming field)
  //
  // The operator matrix is split in two factors for backward stability: GetMat_UC2DE0() and GetMat_UC2DE1().
  // If we multiply the two matrices together then we lose accuracy.

  static const sctl::Matrix<Real>& GetMat_UC2DE0();
  static const sctl::Matrix<Real>& GetMat_UC2DE1();

};

#include "periodize.cpp"

#endif
