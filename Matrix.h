#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <iostream>

template<class T>
class Matrix
{
public:
	Matrix(int m=0, int c=0)
		: m_row(m)
		, m_column(c)
		, m_data(NULL)
	{
		if(m*c > 0)
		{
			m_data = new T[m*c];
		}
	}

	~Matrix()
	{
		delete[] m_data;
		m_data = NULL;
	}

	void resize(int m, int n)
	{
		m_row = m;
		m_column = n;
		if(m_data != NULL)
		{
			delete[] m_data;
		}
		m_data = new T[m*n];
	}

	T& get(int row, int col)
	{
		return m_data[row*m_column+col];
	}

	void set(int row, int col, T value)
	{
		m_data[row*m_column+col] = value;
	}

private:
	T *m_data;
	int m_row;
	int m_column;
};


#endif // __MATRIX_H__
