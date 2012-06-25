#ifndef __matrix44f_h__
#define __matrix44f_h__

class c_matrix44f
{
public:
	c_matrix44f(void);

	c_matrix44f(float mat[4][4]);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// 
	/// \param	m00	Value for entry (0, 0). 
	/// \param	m01	Value for entry (0, 1). 
	/// \param	m02	Value for entry (0, 2). 
	/// \param	m03	Value for entry (0, 3). 
	/// \param	m10	Value for entry (1, 0). 
	/// \param	m11	Value for entry (1, 1). 
	/// \param	m12	Value for entry (1, 2). 
	/// \param	m13	Value for entry (1, 3). 
	/// \param	m20	Value for entry (2, 0). 
	/// \param	m21	Value for entry (2, 1). 
	/// \param	m22	Value for entry (2, 2). 
	/// \param	m23	Value for entry (2, 3). 
	/// \param	m30	Value for entry (3, 0). 
	/// \param	m31	Value for entry (3, 1). 
	/// \param	m32	Value for entry (3, 2). 
	/// \param	m33	Value for entry (3, 3). 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	c_matrix44f(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33);
	~c_matrix44f(void) {}

	// Data members
public:
	/// Entries of the matrix. Public for convenience.
	float m[4][4];

	// Operators
public:
	/// Matix addition operator.
	c_matrix44f operator+(const c_matrix44f& mat) const;
	/// Matrix addition assignment operator.
	c_matrix44f& operator+=(const c_matrix44f& mat);

	/// Matrix subtraction operator.
	c_matrix44f operator-(const c_matrix44f& mat) const;
	/// Matrix subtraction assignment operator.
	c_matrix44f& operator-=(const c_matrix44f& mat);

	/// Matrix scaling operator.
	c_matrix44f operator*(float f) const;
	/// Matrix scaling assignment operator.
	c_matrix44f& operator*=(float f);

	/// Matrix multiplication operator.
	c_matrix44f operator*(const c_matrix44f& mat) const;
	/// Matrix multiplication assignment operator.
	c_matrix44f& operator*=(const c_matrix44f& mat);

public:
	
	float get_elem(int row, int col) const
	{
		assert(row >= 0 && row < 4 && col >= 0 && col < 4);
		return m[row][col];
	}
	
	// Transpose
	c_matrix44f transpose() const;

	c_matrix44f inverse() const;
};

typedef c_matrix44f matrix44f; 

#endif // __matrix44f_h__
