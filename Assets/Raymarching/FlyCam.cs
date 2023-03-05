using UnityEngine;

//------------------------------------------------------------------------------------------------------------------------------------------------------
namespace VolumeRendering
{
    //------------------------------------------------------------------------------------------------------------------------------------
    public class FlyCam : MonoBehaviour
    {
        //--------------------------------------------------------------------------------------------------------------------------------
        public float cameraSensitivity = 90;
        public float normalMoveSpeed = 10;
        public float slowMoveFactor = 0.25f;
        public float fastMoveFactor = 3;

        //--------------------------------------------------------------------------------------------------------------------------------
        private float rotationX = 0.0f;
        private float rotationY = 0.0f;

        //--------------------------------------------------------------------------------------------------------------------------------
        void OnEnable()
        {
            rotationX = transform.eulerAngles.y;
            rotationY = -transform.eulerAngles.x;
        }

        //--------------------------------------------------------------------------------------------------------------------------------
        void Update()
        {
            if (Input.GetMouseButton(1))
            {
                rotationX += Input.GetAxis("Mouse X") * cameraSensitivity * Time.deltaTime;
                rotationY += Input.GetAxis("Mouse Y") * cameraSensitivity * Time.deltaTime;
                rotationY = Mathf.Clamp(rotationY, -90, 90);
            }

            transform.localRotation = Quaternion.AngleAxis(rotationX, Vector3.up);
            transform.localRotation *= Quaternion.AngleAxis(rotationY, Vector3.left);

            float speed = normalMoveSpeed;
            if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift))
            {
                speed *= fastMoveFactor;
            }
            else if (Input.GetKey(KeyCode.LeftControl))
            {
                speed *= slowMoveFactor;
            }

            transform.position += transform.forward * speed * Input.GetAxis("Vertical") * Time.deltaTime;
            transform.position += transform.right * speed * Input.GetAxis("Horizontal") * Time.deltaTime;

            if (Input.GetKey(KeyCode.Q))
            {
                transform.position -= transform.up * speed * Time.deltaTime;
            }
            if (Input.GetKey(KeyCode.E))
            {
                transform.position += transform.up * speed * Time.deltaTime;
            }
        }
    }
}